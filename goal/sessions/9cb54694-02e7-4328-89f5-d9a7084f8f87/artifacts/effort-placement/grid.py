from common import *
PROMPTS={
 'arith':   ('What is 17*23 + 48/6? Reply with just the number.', lambda a: '399' in a),
 'fact':    ('What is the capital of Australia? One word.', lambda a: 'canberra' in a.lower()),
 'code':    ('Write a Python function is_prime(n) that returns True for primes and False otherwise, handling n<2. Return only the code in a ```python block.', lambda a: 'def is_prime' in a and 'return' in a),
 'logic':   ('Alice is older than Bob. Bob is older than Carol. Dave is younger than Carol. Who is the second youngest? Reply with just the name.', lambda a: 'carol' in a.lower()),
 'math':    ('How many positive integers less than 1000 are divisible by 7 but not by 5? Reply with just the number.', lambda a: '114' in a),
 'prose':   ('Write a two-sentence product description for a stainless steel water bottle.', lambda a: len(a.split())>=12),
 'edit':    ('Fix the bug in this function and return only the corrected code:\n```python\ndef mean(xs):\n    return sum(xs) / len(xs) if xs else 0\n\ndef median(xs):\n    s = sorted(xs)\n    n = len(s)\n    return s[n//2] if n % 2 else (s[n//2] + s[n//2+1]) / 2\n```', lambda a: 's[n//2-1]' in a.replace(' ','') or 's[n//2-1]' in a),
}
CONFIGS=[('xhigh','system'),('low','system'),('medium','-'),('xhigh','tail_user'),('low','tail_user')]
def run(name, prompt, check, eff, place):
    if place=='tail_user':
        content=prompt+'\n\n'+INSTR[eff]; effort='medium'
    else:
        content=prompt; effort=eff
    body={'model':'Qwen3.8-27B','messages':[{'role':'user','content':content}],'reasoning_effort':effort,'temperature':0,'max_tokens':6144,'seed':1}
    t=time.time(); r=post('/v1/chat/completions',body); dt=time.time()-t
    m=r['choices'][0]['message']; rc=m.get('reasoning_content') or m.get('reasoning') or ''; ans=m.get('content') or ''
    out={'prompt':name,'effort':eff,'placement':place,'reason_tokens':ntok(rc),'completion_tokens':r['usage']['completion_tokens'],'prompt_tokens':r['usage']['prompt_tokens'],'finish':r['choices'][0]['finish_reason'],'ok':bool(check(ans)),'secs':round(dt,1),'answer':ans[:300],'reasoning_head':rc[:200]}
    print(json.dumps({k:v for k,v in out.items() if k not in('answer','reasoning_head')}), flush=True)
    return out
jobs=[(n,p,c,e,pl) for n,(p,c) in PROMPTS.items() for e,pl in CONFIGS]
res=[]
with cf.ThreadPoolExecutor(4) as ex:
    for o in ex.map(lambda j: run(*j), jobs): res.append(o)
json.dump(res, open('grid.json','w'), indent=1)
