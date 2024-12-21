import azure.cognitiveservices.speech as speechsdk
import json
import string
import time
import threading
import wave
import utils
import json
import fire
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from pprint import pprint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Use the API_KEY from the .env file
SPEECH_KEY = os.getenv('SPEECH_KEY')  # Use the API_KEY from the .env file
print(OPENAI_API_KEY)

DB_FAISS_PATH = 'db_faiss'
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large")

speech_key, service_region = SPEECH_KEY, "westus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

def get_speaking_assessment(audio_file_path):
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language="en-US", audio_config=audio_config)

    pronunciation_config = speechsdk.PronunciationAssessmentConfig( 
            reference_text="", 
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark, 
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme, 
            enable_miscue=False) 
    pronunciation_config.enable_prosody_assessment() 
    # pronunciation_config.enable_content_assessment_with_topic("greeting") WILL ADD

    speech_recognizer.session_started.connect(lambda evt: print(f"SESSION ID: {evt.session_id}"))
    pronunciation_config.apply_to(speech_recognizer)
    speech_recognition_result = speech_recognizer.recognize_once()

    # The pronunciation assessment result as a Speech SDK object
    #pronunciation_assessment_result = speechsdk.PronunciationAssessmentResult(speech_recognition_result)

    # The pronunciation assessment result as a JSON string
    pronunciation_assessment_result_json = speech_recognition_result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)

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

    parsed_assessment_result_data = parse_assessment_result_json(pronunciation_assessment_result_json)

    return parsed_assessment_result_data

def give_feedbacks(llm, parsed_assessment_result_data, criteria):
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

    formatted_prompt = prompt.format(criteria=criteria, parsed_data=parsed_assessment_result_data)
    feedback_response = llm.invoke(formatted_prompt).content

    return feedback_response

def give_feedbacks(parsed_data, llm, criteria, vectorstore):
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
        #f"Here is the exam name: {criteria}"
        # "the question. If you don't know the answer, say that you "
        # "don't know. Use three sentences maximum and keep the "
        # "answer concise."
        # "\n\n"
        )
    human_prompt = (
            f"{parsed_data}"
            f"Use the provided data extracted from a speaking session to provide feedback on the user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy. The user is taking this exam {criteria}. Given the exam provided, use the exam's rubric and point out specific instance where the users can lose points according to the exam's critieria from the system prompt and how they can improve. Do not mention the accuracy score from the provided data, but convert the accuracy score to the score from the exam's rubric. Give the overall score as well as the score for each section (user's pronunciation, fluency, coherence, lexical resource, grammatical range and accuracy) according to the criteria provided in the system prompt."
            
            # "the question. If you don't know the answer, say that you "
            # "don't know. Use three sentences maximum and keep the "
            # "answer concise."
            # "\n\n"
            # "{context}"
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
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever
    # )

    # response = qa.invoke(query)
    # return response["result"]


def main(audio_file_path, criteria, model="gpt-4o"):
    llm = ChatOpenAI(
        model=model,
        api_key="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", 
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    parsed_assessment_result_data = get_speaking_assessment(audio_file_path)
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    feedback_response = give_feedbacks(parsed_data=parsed_assessment_result_data, llm=llm, criteria=criteria, vectorstore=vectorstore)
    print(feedback_response)

if __name__ == "__main__":
    fire.Fire(main)