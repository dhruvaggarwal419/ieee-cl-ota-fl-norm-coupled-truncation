import sys,requests,urllib.parse
qs=open(sys.argv[1]).read().strip().split('\n')
for q in qs:
    import time
    for _ in range(5):
        resp=requests.get('https://api.crossref.org/works',params={'query.bibliographic':q,'rows':2,'select':'DOI,title,author,container-title,issued,type'},timeout=30)
        if resp.status_code==200: break
        time.sleep(3)
    r=resp.json()['message']['items']
    time.sleep(1)
    print('Q:',q)
    for it in r:
        au=', '.join(a.get('family','?') for a in it.get('author',[])[:4])
        print('  ',it['DOI'],'|',it['title'][0][:90],'|',au,'|',(it.get('container-title') or [''])[0][:50],'|',it['issued']['date-parts'][0][0])
