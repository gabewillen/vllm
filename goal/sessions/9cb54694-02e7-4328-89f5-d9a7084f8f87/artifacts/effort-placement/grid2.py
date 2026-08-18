from common import *
import concurrent.futures as cf, json, time
PROMPTS={
 'aime': ('Find the number of ordered pairs of positive integers (a,b) with a<=b such that lcm(a,b)=2^3*3^2*5. Reply with just the number.', lambda a: '18' in a.strip().split()[-1] if a.strip() else False),
 'count': ('How many 5-digit positive integers have digits that are strictly increasing from left to right? Reply with just the number.', lambda a: '126' in a),
 'algo': ('Write a Python function longest_valid_parentheses(s: str) -> int returning the length of the longest well-formed parentheses substring, in O(n) time and O(1) extra space (no stack). Include a docstring. Return only code in one ```python block.', lambda a: 'def longest_valid_parentheses' in a and 'stack' not in a.lower().split('def longest_valid_parentheses')[1][:2000].replace('no stack','')),
 'debug': ('This function should return the k most frequent elements but has bugs. Return only the fixed code in a ```python block.\n```python\nimport heapq\nfrom collections import Counter\n\ndef top_k(nums, k):\n    cnt = Counter(nums)\n    heap = []\n    for x, c in cnt.items():\n        heapq.heappush(heap, (c, x))\n        if len(heap) > k:\n            heapq.heappop(heap)\n    return [x for c, x in heap]\n```\nHint: consider ordering of the result (most frequent first) and ties by smaller value first.', lambda a: 'top_k' in a and ('sorted' in a or 'heappop' in a)),
 'puzzle': ('Five houses in a row are painted red, blue, green, yellow, white (one each). The green house is immediately left of the white house. The red house is in the middle. The blue house is not at either end. The yellow house is somewhere left of the blue house. Which color is the first house? Reply with just the color.', lambda a: 'yellow' in a.lower()),
}
CONFIGS=[('xhigh','system'),('low','system'),('medium','-'),('xhigh','tail_user'),('low','tail_user')]
def run(name, prompt, check, eff, place):
    if place=='tail_user': content=prompt+'\n\n'+INSTR[eff]; effort='medium'
    else: content=prompt; effort=eff
    body={'model':'Qwen3.8-27B','messages':[{'role':'user','content':content}],'reasoning_effort':effort,'temperature':0,'max_tokens':20000,'seed':1}
    t=time.time(); r=post('/v1/chat/completions',body,timeout=1800); dt=time.time()-t
    m=r['choices'][0]['message']; rc=m.get('reasoning_content') or m.get('reasoning') or ''; ans=m.get('content') or ''
    out={'prompt':name,'effort':eff,'placement':place,'reason_tokens':ntok(rc),'completion_tokens':r['usage']['completion_tokens'],'finish':r['choices'][0]['finish_reason'],'ok':bool(check(ans)),'secs':round(dt,1),'answer':ans[:400]}
    print(json.dumps({k:v for k,v in out.items() if k!='answer'}), flush=True); return out
jobs=[(n,p,c,e,pl) for n,(p,c) in PROMPTS.items() for e,pl in CONFIGS]
res=[]
with cf.ThreadPoolExecutor(5) as ex:
    for o in ex.map(lambda j: run(*j), jobs): res.append(o)
json.dump(res, open('grid2.json','w'), indent=1)
