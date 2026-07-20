package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"charm.land/catwalk/pkg/catwalk"
)

type ChatCompletionRequest struct {
	Model                 string         `json:"model"`
	Messages              []ChatMessage  `json:"messages"`
	ReasoningBudgetTokens int64          `json:"reasoning_budget_tokens,omitempty"`
	ThinkingBudgetTokens  int64          `json:"thinking_budget_tokens,omitempty"`
	ChatTemplateKwargs    map[string]any `json:"chat_template_kwargs,omitempty"`
	MaxTokens             int64          `json:"max_tokens,omitempty"`
	MaxCompletionTokens   int64          `json:"max_completion_tokens,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ImageContent struct {
	Type     string   `json:"type"`
	ImageURL ImageURL `json:"image_url"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type ChatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	promptFile := flag.String("p", "", "Path to the text file containing the prompt")
	modelID := flag.String("m", "llava", "Model ID to use")
	apiURL := flag.String("u", "http://localhost:11434/v1", "Base URL of the API")
	reasoning := flag.String("r", "off", "Reasoning level: off, low, medium, high, unlimited")
	showThinking := flag.Bool("t", false, "Display thinking on stdout")
	skipExisting := flag.Bool("s", false, "Skip processing if the .txt file already exists")
	flag.Parse()

	if *promptFile == "" || len(flag.Args()) == 0 {
		fmt.Println("Usage: catwalk-vision -p prompt.txt [-r level] [-t] [-s] <image_glob_or_files>")
		os.Exit(1)
	}

	var budget int64 = 0
	switch strings.ToLower(*reasoning) {
	case "low":
		budget = 512
	case "medium":
		budget = 2048
	case "high":
		budget = 8192
	case "unlimited":
		budget = -1
	}

	promptData, err := os.ReadFile(*promptFile)
	if err != nil {
		fmt.Printf("Error reading prompt file: %v\n", err)
		os.Exit(1)
	}
	prompt := string(bytes.TrimSpace(promptData))

	provider := catwalk.Provider{
		ID:          catwalk.InferenceProvider("local"),
		APIEndpoint: *apiURL,
	}

	var files []string
	for _, arg := range flag.Args() {
		matches, err := filepath.Glob(arg)
		if err != nil || len(matches) == 0 {
			files = append(files, arg)
			continue
		}
		files = append(files, matches...)
	}

	for _, imgPath := range files {
		if !isImage(imgPath) {
			continue
		}

		// Calculate output path
		outPath := strings.TrimSuffix(imgPath, filepath.Ext(imgPath)) + ".txt"

		// Check if we should skip
		if *skipExisting {
			if _, err := os.Stat(outPath); err == nil {
				fmt.Printf("Skipping %s (txt already exists)\n", imgPath)
				continue
			}
		}

		fmt.Printf("--- Processing: %s ---\n", imgPath)
		content, thinking, err := queryLLM(provider, *modelID, prompt, imgPath, budget)
		if err != nil {
			fmt.Printf("Error processing %s: %v\n", imgPath, err)
			continue
		}

		if *showThinking && thinking != "" {
			fmt.Printf("\033[2m[Thinking]: %s\033[0m\n\n", thinking)
		}

		if err := os.WriteFile(outPath, []byte(content), 0644); err != nil {
			fmt.Printf("Error saving %s: %v\n", outPath, err)
			continue
		}
		fmt.Printf("Result saved to: %s\n", outPath)
	}
}

func isImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp"
}

func queryLLM(p catwalk.Provider, model, prompt, imgPath string, budget int64) (string, string, error) {
	imgData, err := os.ReadFile(imgPath)
	if err != nil {
		return "", "", err
	}

	base64Img := base64.StdEncoding.EncodeToString(imgData)
	dataURL := fmt.Sprintf("data:image/jpeg;base64,%s", base64Img)

	reqBody := ChatCompletionRequest{
		Model: model,
		Messages: []ChatMessage{{
			Role: "user",
			Content: []any{
				TextContent{Type: "text", Text: prompt},
				ImageContent{Type: "image_url", ImageURL: ImageURL{URL: dataURL}},
			},
		}},
		ChatTemplateKwargs: make(map[string]any),
	}

	if budget != 0 {
		reqBody.ReasoningBudgetTokens = budget
		reqBody.ThinkingBudgetTokens = budget
		reqBody.ChatTemplateKwargs["enable_thinking"] = true
		if budget > 0 {
			reqBody.MaxTokens = budget + 4096
		} else {
			reqBody.MaxTokens = 32768
		}
	} else {
		reqBody.ReasoningBudgetTokens = 0
		reqBody.ChatTemplateKwargs["enable_thinking"] = false
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", "", err
	}

	url := strings.TrimSuffix(p.APIEndpoint, "/") + "/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", "", fmt.Errorf("API error (%d): %s", resp.StatusCode, string(body))
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", err
	}

	if len(result.Choices) == 0 {
		return "", "", fmt.Errorf("no choices returned")
	}

	content := result.Choices[0].Message.Content
	thinking := result.Choices[0].Message.ReasoningContent

	if thinking == "" {
		re := regexp.MustCompile(`(?s)<think>(.*?)</think>`)
		if match := re.FindStringSubmatch(content); len(match) > 1 {
			thinking = strings.TrimSpace(match[1])
			content = strings.TrimSpace(re.ReplaceAllString(content, ""))
		}
	}

	return content, thinking, nil
}
