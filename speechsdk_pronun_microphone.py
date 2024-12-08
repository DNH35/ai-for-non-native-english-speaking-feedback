import azure.cognitiveservices.speech as speechsdk
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts.criteria import criteria

# Load environment variables from .env file
load_dotenv()

# Get API keys and configuration
SPEECH_KEY = os.getenv('SPEECH_KEY')
OPENAI_API_KEY = "s" + os.getenv('OPENAI_API_KEY')
speech_key, service_region = SPEECH_KEY, "westus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000")

def pronunciation_assessment_from_microphone():
    """Real-time pronunciation assessment with microphone input."""
    # Create microphone configuration
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US")
    pronunciation_config.apply_to(recognizer)
    
    print("READY TO SPEAK. PLEASE SPEAK INTO THE MICROPHONE.")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Recognized Speech: {result.text}")
        print("Pronunciation Assessment Results:")
        pronunciation_result_json = result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        assessment_data = parse_assessment_result_json(pronunciation_result_json)
        feedback = give_feedbacks(setup_llm(), assessment_data)
        print(feedback)

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech was recognized. Please try again.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech recognition was canceled.")
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")


def parse_assessment_result_json(pronunciation_assessment_result_json):
    result_json_data = json.loads(pronunciation_assessment_result_json)
    parsed_data = {
        "Id": result_json_data.get("Id"),
        "Recognition Status": result_json_data.get("RecognitionStatus"),
        "Text Analysis": {
            "Text": result_json_data.get("DisplayText"),
            "Duration (ms)": result_json_data.get("Duration"),
            "Confidence": result_json_data.get("NBest", [{}])[0].get("Confidence"),
        },
        "Pronunciation Assessment": {
            "Scores": result_json_data.get("NBest", [{}])[0].get("PronunciationAssessment"),
        },
        "Detailed Word Analysis": []
    }
    
    for word in result_json_data.get("NBest", [{}])[0].get("Words", []):
        word_info = {
            "Word": word.get("Word"),
            "Offset (ms)": word.get("Offset"),
            "Duration (ms)": word.get("Duration"),
            "Accuracy Score": word.get("PronunciationAssessment", {}).get("AccuracyScore"),
            "Phonemes": [
                {
                    "Phoneme": phoneme.get("Phoneme"),
                    "Accuracy Score": phoneme.get("PronunciationAssessment", {}).get("AccuracyScore"),
                    "Offset (ms)": phoneme.get("Offset"),
                    "Duration (ms)": phoneme.get("Duration"),
                }
                for phoneme in word.get("Phonemes", [])
            ],
        }
        parsed_data["Detailed Word Analysis"].append(word_info)
    
    return parsed_data

def give_feedbacks(llm, parsed_data):
    """Generates feedback using the LLM model."""
    # system_prompt = (
    #     "You are a knowledgeable assistant helping non-native English speakers improve their pronunciation. "
    #     "Use the provided criteria to evaluate the user's performance and offer constructive feedback."
    #     "{criteria}"
    # )

    # human_prompt = (
    #     "Here is the data from a speaking session. Provide feedback on pronunciation, fluency, coherence, lexical resource, "
    #     "grammatical range, and accuracy. Include suggestions for improvement and provide an overall score:"
    #     "{parsed_data}"
    # )
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", system_prompt),
    #     ("human", human_prompt),
    # ])
    # formatted_prompt = prompt.format(criteria=criteria, parsed_data=parsed_data)
    # return llm.invoke(formatted_prompt).content
    system_prompt = (
        "You are a knowledgeable, helpful assistant to help non-native English speakers to improve their English-speaking skills"
        "Use the criteria to grade the pronunciation of the user's speech. Provide feedback on the user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy."
        "{criteria}"
        # "the question. If you don't know the answer, say that you "
        # "don't know. Use three sentences maximum and keep the "
        # "answer concise."
        # "\n\n"
        # "{context}"
            )

    human_prompt = (
        "Use the following data extracted from a speaking session to provide feedback on the user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy. Point out specific instance where the users can lose points according to the critieria and how they can improve. Give the overall score according to the criteria provided in the system prompt."
        "{parsed_data}"
        # "the question. If you don't know the answer, say that you "
        # "don't know. Use three sentences maximum and keep the "
        # "answer concise."
        # "\n\n"
        # "{context}"
            )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    formatted_prompt = prompt.format(criteria=criteria, parsed_data=parsed_data)
    feedback_response = llm.invoke(formatted_prompt).content

    return feedback_response

def setup_llm(model="gpt-4"):
    """Initializes the LLM for feedback generation."""
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )

if __name__ == "__main__":
    pronunciation_assessment_from_microphone()
