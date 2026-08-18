import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); same=0
for k in a:
    ta=(a[k]["reasoning"] or "")+(a[k]["content"] or ""); tb=(b[k]["reasoning"] or "")+(b[k]["content"] or "")
    eq = ta==tb; same+=eq
    if not eq:
        i=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y), min(len(ta),len(tb)))
        print(f"{k}: DIFF at char {i}/{len(ta)} ({a[k]['usage']['completion_tokens']} vs {b[k]['usage']['completion_tokens']} tok)")
    else: print(f"{k}: identical ({a[k]['usage']['completion_tokens']} tok)")
print(f"identical {same}/{len(a)}")
