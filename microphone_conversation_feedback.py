import azure.cognitiveservices.speech as speechsdk
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from flask import Flask, request, jsonify
import fire 


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

SPEECH_KEY = os.getenv('SPEECH_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
speech_key, service_region = SPEECH_KEY, "westus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)



embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")

def pronunciation_assessment_from_microphone(criteria, previous_feedback="", vectorstore_path = ""):
    """Real-time pronunciation assessment with microphone input."""
    # Create microphone configuration
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()

    
    speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000")

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US")
    pronunciation_config.apply_to(recognizer)
    
    print("READY TO SPEAK. PLEASE SPEAK INTO THE MICROPHONE.")

    result = recognizer.recognize_once_async().get()

    # print("RESULT REASON:")

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Recognized Speech: {result.text}")
        print("Pronunciation Assessment Results:")
        pronunciation_result_json = result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        assessment_data = parse_assessment_result_json(pronunciation_result_json)
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        feedback = give_feedbacks(parsed_data=assessment_data, llm=setup_llm(max_tokens=1024*4), criteria=criteria, prev_feedback=previous_feedback, vectorstore=vectorstore)
        return feedback

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech was recognized. Please try again.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech recognition was canceled.")
        cancellation_details = result.cancellation_details
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")

    return None


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

def give_feedbacks(parsed_data, llm, criteria, prev_feedback,  vectorstore):
    '''
    Query the retrieval chain with the given query and vector store

    Parameters:
        query (str): The query to search for
        vectorstore (FAISS): The vector store to search in

    Returns:
        str: The response from the retrieval chain
    '''
    similar_embeddings = vectorstore.similarity_search(criteria)
    similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=embedding_model)

    

    #retriever = vectorstore.as_retriever()
    retriever = similar_embeddings.as_retriever()
    system_prompt = (
        "You are a knowledgeable, helpful assistant to help non-native English speakers to improve their English-speaking skills\n"
        "Use the exam name to grade the pronunciation of the user's speech according to the exam's criteria in the vector database, provide feedback on the user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy. Give score for each of the categories and the overall score according to the exam's criteria."
        )
    
    human_prompt = ""
    
    if prev_feedback == "":
        human_prompt = (
                f"{parsed_data}"
                f"Use the provided data extracted from a speaking session to provide feedback on the user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy. The user is taking this exam {criteria}. Given the exam provided, use the exam's rubric and point out specific instance where the users can lose points according to the exam's critieria from the system prompt and how they can improve. Do not mention the accuracy score from the provided data, but convert the accuracy score to the score from the exam's rubric. Give the overall score as well as the score for each section (user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy) according to the criteria provided in the system prompt."
                    )
        
    else:
        human_prompt = (
                f"{parsed_data}"
                f"Given the previous feedback and score that the user already has, update the score and feedback given the new data extracted from a speaking session. The user is taking this exam {criteria}. Use the exam's rubric and point out specific instance where the users can lose points according to the exam's critieria from the system prompt and how they can improve. Do not mention the accuracy score from the provided data, but convert the accuracy score to the score from the exam's rubric. Give the overall score as well as the score for each section (user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy) according to the criteria provided in the system prompt."
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{context}"),
            ("human", "{input}"),
        ]
    )
    print("SYSTEM PROMPT: ", system_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    #include context to be system prompt
    results = rag_chain.invoke({"context": system_prompt, "input": human_prompt})
    print("RESULTS: ", results)
    # sources = set([doc.metadata.get('source') for doc in results['context']])
    return results['answer']

def setup_llm(model="gpt-4", temperature=0.2, top_p=0.7, max_tokens=1024):
    """Initializes the LLM for feedback generation."""
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

def generate_questions(sample_questions, model):
    system_prompt = "You are an examiner for an english international exam that has a lot of knowledge in a wide range of topics. You have experience working with non-native speakers. You are asked to come up with new questions and topics for the next exam."
    human_prompt = f"""Given the following questions and topics, come up with one more unique unique topic and ask a list of conversational questions relating to that topic for the next exam. Here are the questions: \n\n{sample_questions}
    Your response should be in the following format:
    Topic: [Your topic here]
    Questions: [Your list of questions here]
    """

    prompt = [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    
    llm = setup_llm(model, temperature=0.5)

    question_responses = llm.invoke(prompt).content
    questions_responses_arr = question_responses.split("\n")[3:] #hard coded

    return questions_responses_arr

def generate_audio_for_questions(question):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    speech_synthesizer.speak_text_async(question).get()

def main(criteria="IELTS", sample_questions_path="/Users/duynguyen/ai-for-non-native-english-speaking-feedback/data/ietls_speaking_questions.txt", vectorstore_path="/Users/duynguyen/ai-for-non-native-english-speaking-feedback/db_faiss"):
    question_file_content = open(sample_questions_path, 'r')

    generated_questions_arr = generate_questions(question_file_content.read(), "gpt-4o")
    feedback_lst = []
    prev_feedback = ""
    feedback = ""
    for i, question in enumerate(generated_questions_arr):
        print("QUESTION: ", question)
        generate_audio_for_questions(question)
        feedback = pronunciation_assessment_from_microphone(criteria, prev_feedback, vectorstore_path)
        print(f"FEEDBACK FOR QUESTION {i + 1}: {feedback}")
        feedback_lst.append(feedback)
        prev_feedback = feedback


    
        
if __name__ == "__main__":
    fire.Fire(main)


    
