import base64
import requests
import json
import os

# Base64 encoding format
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# Prepare request
api_key = "sk-fbfbf8f489a043ac86e6ec65586c34dd"
# Use OpenAI compatible interface
url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Create a folder to store Markdown files
output_folder = "analysis_results"
os.makedirs(output_folder, exist_ok=True)

# Get all video files from square_videos folder
video_folder = "square_videos"
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Supported video formats

# Check if folder exists
if not os.path.exists(video_folder):
    print(f"Error: '{video_folder}' folder does not exist")
    exit(1)

# Get all video file paths
video_files = []
for file in os.listdir(video_folder):
    file_path = os.path.join(video_folder, file)
    if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in video_extensions:
        video_files.append(file_path)

if not video_files:
    print(f"Error: No video files found in '{video_folder}' folder")
    exit(1)

print(f"Found {len(video_files)} video files, starting processing...")

# Process each video file
for video_path in video_files:
    print(f"\nProcessing video: {video_path}")
    
    # Encode video file to Base64
    try:
        base64_video = encode_video(video_path)
    except Exception as e:
        print(f"Failed to read video file: {e}")
        continue
    
    data = {
        "model": "qwen-vl-max",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a professional traffic accident analysis expert, skilled in assessing collision risks through vehicle trajectory data. When analyzing videos, please note: 1) The blue and red circles represent two different vehicles; 2) Observe the movement direction, speed changes, and closest approach distance of the two vehicles; 3) Consider the actual size of the vehicles (not just the circle size); 4) Analyze whether there are intersecting paths or simultaneous occupation of the same space.\n\nPlease organize your response using clear titles, sections, bullet points, and tables to make the analysis results well-structured and easy to read. Use bold to highlight key conclusions and data. The conclusion section should succinctly summarize the analysis results."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
                    },
                    {"type": "text", "text": "Please analyze this vehicle trajectory animation and present the results in a professional and aesthetically pleasing format:\n\n## Analysis Points\n1. Is there a collision risk between the two vehicles (marked in blue and red)?\n2. Please identify key time points (e.g., when the closest distance occurs)\n3. Analyze whether there is any evasive or acceleration behavior based on the vehicle movement patterns\n4. Provide your professional judgment and reasoning\n\nPlease use markdown format, and use titles, tables, bullet points, and bold elements appropriately to make the response clearer and more readable."}
                ]
            }
        ],
        "parameters": {"fps": 9}
    }

    # Extract video filename (without extension)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    try:
        # Send request
        print(f"Sending API request...")
        response = requests.post(url, headers=headers, json=data)
        
        # Print response status code
        print(f"Status code: {response.status_code}")
        
        # Parse JSON response
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis successful, saving results...")
            analysis_content = result["choices"][0]["message"]["content"]
            
            # Write analysis result to Markdown file
            md_file_path = os.path.join(output_folder, f"{video_filename}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(analysis_content)
            print(f"Analysis results saved to: {md_file_path}")
        else:
            print(f"Request failed, status code: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except Exception as e:
        print(f"Error occurred while processing video: {e}")

print(f"\nAll videos processed. Results saved in '{output_folder}' folder.")