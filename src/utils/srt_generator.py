import datetime

def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds (float)
        
    Returns:
        String formatted as HH:MM:SS,mmm
    """
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)
    
    # Get total seconds
    total_seconds = int(td.total_seconds())
    
    # Calculate hours, minutes, seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    # Calculate milliseconds
    milliseconds = int((seconds - total_seconds) * 1000)
    
    # Format the string
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def generate_srt(segments: list, output_path: str) -> str:
    """
    Creates an SRT file from segments with 'start', 'end', and 'translated_text' keys.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path to write the SRT file
        
    Returns:
        Path to the generated SRT file
    """
    srt_content = []
    
    for i, seg in enumerate(segments, start=1):
        start_time = format_timestamp(seg.get('start', 0.0))
        end_time = format_timestamp(seg.get('end', 0.0))
        text = seg.get('translated_text', '').strip()
        
        if not text:
            continue
            
        srt_content.append(f"{i}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")  # Empty line after each block
        
    # Join with newlines
    content = "\n".join(srt_content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return output_path
