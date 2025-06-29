"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""


import base64
import math
import os
import tempfile
from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import whisper
import cv2
import yt_dlp
from moviepy import VideoFileClip

from react_agent.configuration import Configuration
from react_agent.utils import load_chat_model

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import numexpr
from langchain_unstructured import UnstructuredLoader

async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information about a specific topic or query.
    
    This tool uses Wikipedia's API to search for and retrieve information about
    topics, people, places, events, and other subjects. It provides detailed
    information from Wikipedia articles.
    
    Args:
        query: The search term or topic to look up on Wikipedia
        
    Returns:
        Wikipedia article content and information about the searched topic
        
    Examples:
        "Albert Einstein" for information about the physicist
        "Python programming language" for information about Python
        "Machine learning" for information about ML
    """
    try:
        # Create Wikipedia API wrapper
        wikipedia = WikipediaAPIWrapper()
        
        # Create Wikipedia query tool
        wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        
        # Execute the search
        result = wiki_tool.run(query)
        
        return result
        
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# @tool
# def calculator(expression: str) -> str:
#     """Calculate mathematical expression.

#     Expression should be a single line mathematical expression
#     that solves the problem.

#     Examples:
#         "37593 * 67" for "37593 times 67"
#         "37593**(1/5)" for "37593^(1/5)"
#     """
#     local_dict = {"pi": math.pi, "e": math.e}
#     return str(
#         numexpr.evaluate(
#             expression.strip(),
#             global_dict={},  # restrict access to globals
#             local_dict=local_dict,  # add common mathematical functions
#         )
#     )

@tool
def unstructured_file_loader(file_path: str):
    """Load and process unstructured files (PDFs, docs, xlsx, images,.) into text documents.
    
    This tool is to extract text content from various
    file formats including PDFs, Word documents, Excel files, images with text,
    and other unstructured file types.
    
    Args:
        file_path: The path to the file to be loaded and processed
        
    Returns:
        A list of Document objects containing the extracted text content from the file
        
    Examples:
        "load_document.pdf" for a PDF file
        "report.docx" for a Word document
        "reports.xlsx" for an Excel file
    """
    loader = UnstructuredLoader(file_path)
    document = loader.load()

    return document

@tool
def image_analysis(file_path: str, question: str) -> str:
    """Analyze an image and answer a question about it.
    
    This tool loads an image file, converts it to base64, and uses a vision-capable
    language model to answer questions about the image content.
    
    Args:
        file_path: The path to the image file to analyze
        question: The question to ask about the image
        
    Returns:
        The model's answer to the question about the image
        
    Examples:
        "image.jpg" and "What objects do you see in this image?"
        "screenshot.png" and "What text is visible in this image?"
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."
    
    # Read the image file and encode as base64
    try:
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        return f"Error reading image file: {str(e)}"
    
    # Determine file extension to set proper MIME type
    file_extension = os.path.splitext(file_path)[1].lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_type_map.get(file_extension, 'image/jpeg')
    
    # Create the image URL with base64 data
    image_url = f"data:{mime_type};base64,{base64_image}"
    
    # Load the chat model
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Create a message with the image and question
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    )
    
    # Get the model's response
    try:
        response = model.invoke([message])
        return response.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

@tool
def transcribe_media(file_path: str) -> str:
    """Transcribe an audio/video file (MP3, MP4, WAV, etc.) to text.
    Use this tool to answer questions about the spoken content of a video or audio file.

    This tool converts spoken audio and video content
    into text transcription. It supports various audio/video formats including MP3, MP4
    WAV, M4A, and others.
    
    Args:
        file_path: The path to the media file to transcribe
        
    Returns:
        The transcribed text content from the media file
        
    Examples:
        "recording.mp3" for an MP3 audio file
        "recording.mp4" for an MP4 audio file
        "interview.wav" for a WAV audio file
        "podcast.m4a" for an M4A audio file
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."
    
    # Check if file is a supported media file
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    supported_extensions = audio_extensions | video_extensions
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in supported_extensions:
        return f"Error: File '{file_path}' is not a supported media format. Supported formats: {', '.join(supported_extensions)}"
    
    try:
        # If it's a video file, extract audio to WAV first
        if file_extension in video_extensions:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            try:
                # Extract audio from video
                video = VideoFileClip(file_path)
                video.audio.write_audiofile(temp_wav_path)
                video.close()
                
                # Use the temporary WAV file for transcription
                media_path = temp_wav_path
            except Exception as e:
                return f"Error extracting audio from video: {str(e)}"
        else:
            # For audio files, use the original path
            media_path = file_path
        
        # Load the Whisper model (this will download the model on first use)
        model = whisper.load_model("base")
        
        # Transcribe the audio file
        result = model.transcribe(media_path)
        
        # Clean up temporary file if it was created
        if file_extension in video_extensions and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        
        # Return the transcribed text
        return result["text"]
        
    except Exception as e:
        return f"Error transcribing media: {str(e)}"

@tool
def download_youtube_video(youtube_url: str, output_path: Optional[str] = None) -> str:
    """Download a YouTube video to a specified path.
    
    This tool downloads a YouTube video using yt-dlp and saves it to the provided output path.
    If no output path is provided, a random temporary path will be generated.
    
    Args:
        youtube_url: The YouTube URL of the video to download
        output_path: The path where the video should be saved (optional, will generate random path if not provided)
        
    Returns:
        A success message with the path where the video was saved
        
    Examples:
        "https://www.youtube.com/watch?v=example" and "videos/my_video.mp4"
        "https://youtu.be/example" and "/path/to/video.mp4"
        "https://www.youtube.com/watch?v=example" (will generate random path)
    """
    try:
        # Generate random output path if not provided
        if output_path is None:
            import uuid
            random_filename = f"./temp/{uuid.uuid4().hex}.mp4"
            output_path = random_filename
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p to reduce file size
            'outtmpl': output_path,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Check if video was downloaded successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            return f"Successfully downloaded video to '{output_path}' (Size: {file_size} bytes)"
        else:
            return "Error: Failed to download video from the provided URL."
            
    except Exception as e:
        return f"Error downloading video: {str(e)}"

@tool
def video_analysis(video_path: str, question: str) -> str:
    """Analyze a video file and answer a question about the visuals of it.
    Only use this tool to answer questions about the visual content of the video.

    This tool loads a video file, extracts key frames, answers questions about the video content.
    
    Args:
        video_path: The path to the video file to analyze
        question: The question to ask about the visual content of the video
        
    Returns:
        The model's answer to the question about the video
        
    Examples:
        "videos/my_video.mp4" and "What is happening in this video?"
        "/path/to/video.mp4" and "What objects or people do you see?"
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return f"Error: File '{video_path}' not found."
    
    # Check if file is a video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    file_extension = os.path.splitext(video_path)[1].lower()
    
    if file_extension not in video_extensions:
        return f"Error: File '{video_path}' is not a supported video format. Supported formats: {', '.join(video_extensions)}"
    
    try:
        # Extract frames from the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open the video file."
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract frames at regular intervals (every 1 second)
        frame_interval = int(fps) if fps > 0 else 30
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
            
            # Limit to maximum 10 frames to avoid overwhelming the model
            if len(frames) >= 10:
                break
        
        cap.release()
        
        if not frames:
            return "Error: Could not extract any frames from the video."
        
        # Load the chat model
        configuration = Configuration.from_context()
        model = load_chat_model(configuration.model)
        
        # Prepare content for the model
        content = [{"type": "text", "text": f"Question about this video: {question}\n\nI'll show you {len(frames)} key frames from this video to help answer your question."}]
        
        # Add each frame as an image
        for i, frame in enumerate(frames):
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            base64_frame = base64.b64encode(buffer).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_frame}"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # Create a message with the frames and question
        message = HumanMessage(content=content)
        
        # Get the model's response
        response = model.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error analyzing video: {str(e)}"

@tool
def provide_final_answer(summary: str) -> str:
    """Provide a final answer based on the summary of the response.
    
    Always use this tool to provide an answer to the initial  question.

    This tool takes a summary of the research and analysis performed and formats it
    into a final answer according to the specified template. The final answer should be
    concise and follow specific formatting rules.
    
    Args:
        summary: A summary of the research, analysis, and findings
        
    Returns:
        A formatted final answer following the template:
        "YOUR FINAL ANSWER"
        
    Formatting Rules:
    - If asked for a number: don't use commas, don't include units ($, %, etc.) unless specified
    - If asked for a string: don't use articles, don't use abbreviations, write digits in plain text
    - If asked for a comma separated list: apply the above rules to each element
    - Keep the answer as concise as possible with as few words as possible
        
    Examples:
        "What is 2+2?" -> "4"
        "What is the capital of France?" -> "Paris"
        "List the first 3 prime numbers" -> "2, 3, 5"
    """
    # Load the chat model for processing
    configuration = Configuration.from_context()
    model = load_chat_model(configuration.model)
    
    # Create the prompt for formatting the final answer
    prompt = f"""
Based on the following summary of research and analysis, provide a final answer according to these rules:

SUMMARY:
{summary}

RULES FOR FINAL ANSWER:
1. Keep the answer as concise as possible with as few words as possible
2. If asked for a number: don't use commas, don't include units ($, %, etc.) unless specified
3. If asked for a string: don't use articles, don't use abbreviations, write digits in plain text
4. If asked for a comma separated list: apply the above rules to each element

Provide only the final answer value
"""

    # Get the model's response
    try:
        message = HumanMessage(content=prompt)
        response = model.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f"Error formatting {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read the content of a plain text file.
    
    This tool reads and returns the plain text content of files such as .txt, .py, .js, .md,
    .json, .csv, .html, .css, .xml, .yaml, .yml, .toml, .ini, .cfg, .conf, .log, and other
    text-based file formats.
    
    Args:
        file_path: The path to the text file to read
        
    Returns:
        The plain text content of the file
        
    Examples:
        "data.txt" for a text file
        "script.py" for a Python file
        "config.json" for a JSON file
        "README.md" for a markdown file
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."
    
    # Check if file is a text file by extension
    text_extensions = {
        '.txt', '.py', '.js', '.ts', '.md', '.json', '.csv', '.html', '.css', '.xml',
        '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.log', '.sh', '.bash',
        '.zsh', '.fish', '.ps1', '.bat', '.cmd', '.sql', '.r', '.m', '.scala', '.java',
        '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift',
        '.kt', '.dart', '.lua', '.pl', '.pm', '.tcl', '.vbs', '.wsf', '.psm1',
        '.dockerfile', '.gitignore', '.gitattributes', '.editorconfig', '.eslintrc',
        '.prettierrc', '.babelrc', '.webpack.config.js', '.package.json', '.requirements.txt',
        '.setup.py', '.pyproject.toml', '.cargo.toml', '.composer.json', '.gemfile',
        '.pom.xml', '.build.gradle', '.sbt', '.mix.exs', '.rebar.config', '.cabal',
        '.stack.yaml', '.pubspec.yaml', '.go.mod', '.go.sum', '.cargo.lock', '.yarn.lock',
        '.package-lock.json', '.composer.lock', '.gemfile.lock', '.pom.xml.lock',
        '.build.gradle.lock', '.sbt.lock', '.mix.lock', '.rebar.lock', '.cabal.lock',
        '.stack.lock', '.pubspec.lock', '.go.lock', '.cargo.lock', '.yarn.lock',
        '.package-lock.json.lock', '.composer.lock.lock', '.gemfile.lock.lock',
        '.pom.xml.lock.lock', '.build.gradle.lock.lock', '.sbt.lock.lock',
        '.mix.lock.lock', '.rebar.lock.lock', '.cabal.lock.lock', '.stack.lock.lock',
        '.pubspec.lock.lock', '.go.lock.lock', '.cargo.lock.lock', '.yarn.lock.lock'
    }
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension not in text_extensions:
        return f"Error: File '{file_path}' may not be a text file. Supported text extensions: {', '.join(sorted(text_extensions))}"
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return content
        
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"Error reading file with different encoding: {str(e)}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

TOOLS: List[Callable[..., Any]] = [search, wikipedia_search, unstructured_file_loader, image_analysis, transcribe_media, download_youtube_video, video_analysis, provide_final_answer, read_file]
