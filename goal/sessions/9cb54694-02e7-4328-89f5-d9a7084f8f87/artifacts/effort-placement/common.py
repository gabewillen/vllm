import json, os, sys, time, re, urllib.request, concurrent.futures as cf
KEY=os.environ['VLLM_API_KEY']; URL='http://localhost:8012'
INSTR={'xhigh':'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
       'low':'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.'}
def post(path, body, timeout=900):
    r=urllib.request.Request(URL+path, data=json.dumps(body).encode(), headers={'Content-Type':'application/json','Authorization':'Bearer '+KEY})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def ntok(text):
    if not text: return 0
    return post('/tokenize',{'model':'Qwen3.8-27B','prompt':text,'add_special_tokens':False})['count']
