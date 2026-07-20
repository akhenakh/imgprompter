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

	// Importing catwalk for standardized provider and model structures
	"charm.land/catwalk/pkg/catwalk"
)

// ChatCompletionRequest defines the payload structure for OpenAI-compatible APIs.
// It includes specific fields used by llama.cpp for reasoning budgets.
type ChatCompletionRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`

	// ReasoningBudgetTokens is the primary field llama.cpp uses to limit internal "thinking".
	ReasoningBudgetTokens int64 `json:"reasoning_budget_tokens,omitempty"`
	// ThinkingBudgetTokens is an alias often used in various llama.cpp server versions.
	ThinkingBudgetTokens int64 `json:"thinking_budget_tokens,omitempty"`

	// ChatTemplateKwargs allows passing variables to the Jinja template (e.g., enable_thinking).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`

	// MaxTokens defines the total limit for (Reasoning + Final Content).
	MaxTokens           int64 `json:"max_tokens,omitempty"`
	MaxCompletionTokens int64 `json:"max_completion_tokens,omitempty"`
}

// ChatMessage represents a single turn in the conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // Can be string or []any for multi-modal content
}

// TextContent is used within a multi-modal message array.
type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ImageContent is used to pass base64 image data.
type ImageContent struct {
	Type     string   `json:"type"`
	ImageURL ImageURL `json:"image_url"`
}

type ImageURL struct {
	URL string `json:"url"` // Format: "data:image/jpeg;base64,..."
}

// ChatCompletionResponse parses the standardized output from the AI server.
type ChatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"` // Specific field for "thought" text
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	// Define CLI flags
	promptFile := flag.String("p", "", "Path to the text file containing the prompt")
	modelID := flag.String("m", "llava", "Model ID to use")
	apiURL := flag.String("u", "http://localhost:11434/v1", "Base URL of the API")
	reasoning := flag.String("r", "off", "Reasoning level: off, low, medium, high, unlimited")
	showThinking := flag.Bool("t", false, "Display thinking on stdout (not saved to file)")
	skipExisting := flag.Bool("s", false, "Skip processing if the .txt sidecar already exists")
	flag.Parse()

	// Validation
	if *promptFile == "" || len(flag.Args()) == 0 {
		fmt.Println("Usage: catwalk-vision -p prompt.txt [-r level] [-t] [-s] <image_glob_or_files>")
		os.Exit(1)
	}

	// Map reasoning CLI strings to token budgets based on REASONING_EFFORT_TOKENS constants
	var budget int64 = 0
	switch strings.ToLower(*reasoning) {
	case "low":
		budget = 512
	case "medium":
		budget = 2048
	case "high":
		budget = 8192
	case "unlimited":
		budget = -1 // Unlimited tokens
	}

	// Load the text prompt from the user-specified file
	promptData, err := os.ReadFile(*promptFile)
	if err != nil {
		fmt.Printf("Error reading prompt file: %v\n", err)
		os.Exit(1)
	}
	prompt := string(bytes.TrimSpace(promptData))

	// Initialize the catwalk provider structure
	provider := catwalk.Provider{
		ID:          catwalk.InferenceProvider("local"),
		APIEndpoint: *apiURL,
	}

	// Expand globs (e.g., *.jpg) into a flat list of files
	var files []string
	for _, arg := range flag.Args() {
		matches, err := filepath.Glob(arg)
		if err != nil || len(matches) == 0 {
			files = append(files, arg) // Use literal name if glob finds nothing
			continue
		}
		files = append(files, matches...)
	}

	// Iterate through each image file
	for _, imgPath := range files {
		if !isImage(imgPath) {
			continue
		}

		// Determine the sidecar filename (image.png -> image.txt)
		outPath := strings.TrimSuffix(imgPath, filepath.Ext(imgPath)) + ".txt"

		// Handle idempotency check
		if *skipExisting {
			if _, err := os.Stat(outPath); err == nil {
				fmt.Printf("Skipping %s (txt already exists)\n", imgPath)
				continue
			}
		}

		fmt.Printf("--- Processing: %s ---\n", imgPath)

		// Execute the LLM query
		content, thinking, err := queryLLM(provider, *modelID, prompt, imgPath, budget)
		if err != nil {
			fmt.Printf("Error processing %s: %v\n", imgPath, err)
			continue
		}

		// Print reasoning to terminal if requested (dimmed using ANSI 2)
		if *showThinking && thinking != "" {
			fmt.Printf("\033[2m[Thinking]: %s\033[0m\n\n", thinking)
		}

		// Save the final cleaned response to the text file
		if err := os.WriteFile(outPath, []byte(content), 0644); err != nil {
			fmt.Printf("Error saving %s: %v\n", outPath, err)
			continue
		}
		fmt.Printf("Result saved to: %s\n", outPath)
	}
}

// isImage checks extensions for supported image formats
func isImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp"
}

// queryLLM builds the payload and executes the HTTP request
func queryLLM(p catwalk.Provider, model, prompt, imgPath string, budget int64) (string, string, error) {
	// Encode image to Base64
	imgData, err := os.ReadFile(imgPath)
	if err != nil {
		return "", "", err
	}
	base64Img := base64.StdEncoding.EncodeToString(imgData)
	dataURL := fmt.Sprintf("data:image/jpeg;base64,%s", base64Img)

	// Build the request body
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

	// If reasoning is requested, set budget and enable the Jinja template feature
	if budget != 0 {
		reqBody.ReasoningBudgetTokens = budget
		reqBody.ThinkingBudgetTokens = budget
		reqBody.ChatTemplateKwargs["enable_thinking"] = true

		// Set a total token cap high enough for reasoning + response
		if budget > 0 {
			reqBody.MaxTokens = budget + 4096
		} else {
			reqBody.MaxTokens = 32768
		}
	} else {
		// Explicitly disable thinking via Jinja kwarg and set budget to 0
		reqBody.ReasoningBudgetTokens = 0
		reqBody.ChatTemplateKwargs["enable_thinking"] = false
	}

	// Prepare and send the HTTP POST request
	jsonData, _ := json.Marshal(reqBody)
	url := strings.TrimSuffix(p.APIEndpoint, "/") + "/chat/completions"
	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
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

	// Decode the JSON response
	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", err
	}

	if len(result.Choices) == 0 {
		return "", "", fmt.Errorf("no choices returned")
	}

	content := result.Choices[0].Message.Content
	thinking := result.Choices[0].Message.ReasoningContent

	// Fallback: Some models (DeepSeek-R1) embed thinking in content within <think> tags.
	// We extract it so the final file only contains the clean response.
	if thinking == "" {
		re := regexp.MustCompile(`(?s)<think>(.*?)</think>`)
		if match := re.FindStringSubmatch(content); len(match) > 1 {
			thinking = strings.TrimSpace(match[1])
			content = strings.TrimSpace(re.ReplaceAllString(content, ""))
		}
	}

	return content, thinking, nil
}
