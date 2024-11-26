from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio  
from custom_agent_with_caching import generate_response  

app = Flask(__name__)
CORS(app)  


@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    try:
        
        response = await generate_response(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    return '', 204  

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5511, debug=True)


# ollama run llama3.1:70b