import json, os, urllib.request, random, string
NONCE=''.join(random.choices(string.ascii_lowercase,k=8))
KEY=os.environ['VLLM_API_KEY']; URL='http://localhost:8012'
from common import INSTR, post
# a long-ish shared history (~3k tokens) so cache effects are visible
hist=[{'role':'user','content':'Here is a document to remember:\n\n'+('The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. '*400)+'\n\nAcknowledge with OK.'},
      {'role':'assistant','content':'OK.'}]
def metrics():
    import urllib.request
    r=urllib.request.Request(URL+'/metrics',headers={'Authorization':'Bearer '+KEY})
    d={}
    for line in urllib.request.urlopen(r).read().decode().splitlines():
        if line.startswith('vllm:prefix_cache_queries_total') or line.startswith('vllm:prefix_cache_hits_total'):
            d[line.split('{')[0]]=float(line.rsplit(' ',1)[1])
    return d
def turn(eff, place, q, tag):
    m0=metrics()
    if place=='tail_user': msgs=hist+[{'role':'user','content':q+'\n\n'+INSTR[eff]}]; effort='medium'
    else: msgs=hist+[{'role':'user','content':q}]; effort=eff
    r=post('/v1/chat/completions',{'model':'Qwen3.8-27B','messages':msgs,'reasoning_effort':effort,'temperature':0,'max_tokens':64})
    u=r['usage']; m1=metrics()
    q_=m1['vllm:prefix_cache_queries_total']-m0['vllm:prefix_cache_queries_total']; h=m1['vllm:prefix_cache_hits_total']-m0['vllm:prefix_cache_hits_total']
    print(f"{tag:28s} effort={eff:5s} place={place:9s} prompt={u['prompt_tokens']} cache_queried={q_:.0f} cache_hit={h:.0f}")
BASE=hist[0]['content']
for place in ['system','tail_user']:
    hist[0]['content']=BASE
    # unique history per placement so the two runs do not share cache with each other
    hist[0]['content']=('nonce-'+NONCE+'-'+place+'\n')+hist[0]['content'].replace('lazy dog','lazy dog '+NONCE+place)
    turn('xhigh',place,'How many words in the document? Guess.', 'turn1 xhigh (cold)')
    turn('xhigh',place,'How many words in the document? Guess.', 'turn1 xhigh (repeat)')
    turn('low',  place,'How many words in the document? Guess.', 'turn2 switch->low')
    turn('xhigh',place,'Name the animal in the document.',     'turn3 switch->xhigh, new q')
