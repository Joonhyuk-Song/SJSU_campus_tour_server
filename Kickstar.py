import time
from SJSU_RAG import PDFRAGPipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

latest_text = ""
response_text = ""  
pipeline = PDFRAGPipeline(pdf_path=r"C:\Users\joons\Desktop\PES_Project\venv1\Kickstar\SJSU_combined_V1.pdf")
data = pipeline.ingest_pdf()
chunks = pipeline.split_text(data)
pipeline.create_vector_db(chunks)

@app.route('/send_text', methods=['POST'])
def receive_text():
    global latest_text, response_text
    try:
        data = request.get_json()
        latest_text = data.get('text', '')
        print(f"Received text: {latest_text}")
        latest_text = latest_text + " in 30 words"
        start = time.time()
        print("start query")
        response_text = pipeline.query(latest_text)
        finish = time.time()
        print(f"Processing time: {finish - start} seconds")
        return jsonify({"message": "Text received successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Error processing the request", "error": str(e)}), 400

@app.route('/get_text', methods=['GET'])
def get_text():
    global response_text
    if not response_text:
        return jsonify({"message": "No response available yet"}), 400  
    print(f"Response: {response_text}")
    return jsonify({"message": response_text}), 200

if __name__ == '__main__':
    app.run(host='10.251.31.128', port=5000)
