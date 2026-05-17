from pydantic import BaseModel, Field

class TranscriptionResponse(BaseModel):
    transcript: str = Field(..., description="The transcribed text from the audio file")
    filename: str = Field(..., description="The name of the audio file")
    duration_seconds: float = Field(..., description="The duration of the audio file in seconds")