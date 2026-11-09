import requests,time,re,json
dois=[("zhu20","10.1109/twc.2019.2946245"),("yang20","10.1109/twc.2019.2961673"),("amiri20","10.1109/twc.2020.2974748"),
("cao20","10.1109/twc.2020.3012287"),("sery21","10.1109/tsp.2021.3090323"),("abrar24","10.1109/iccworkshops59551.2024.10615488"),
("abrar26","10.1109/twc.2026.3678173"),("sun24","10.1109/twc.2023.3336277"),("zhang21","10.1109/twc.2021.3065748"),
("amiri20b","10.1109/tsp.2020.2981904"),("sery20","10.1109/tsp.2020.2989580"),("comaniciu02","10.1109/34.1000236"),
("corless96","10.1007/bf02124750"),("kairouz21","10.1561/2200000083"),("li20","10.1109/msp.2020.2975749"),
("chen21","10.1109/twc.2020.3024629"),("yang20b","10.1109/tcomm.2019.2944169"),("cao22","10.1109/jsac.2022.3143217"),
("lim20","10.1109/comst.2020.2986024"),("yan25","10.1109/twc.2025.3550159"),("zhu24","10.1109/jiot.2024.3352280"),
("sery20b","10.1109/globecom42002.2020.9322580"),("dinh21","10.1109/tnet.2020.3035770")]
bib=[];chk=["| Key | DOI | Title (Crossref) | Authors (Crossref) | Venue, year, vol, pages | Status |","|---|---|---|---|---|---|"]
for k,d in dois:
    for _ in range(6):
        r=requests.get('https://api.crossref.org/works/'+d,timeout=30)
        if r.status_code==200: break
        time.sleep(3)
    m=r.json()['message']; time.sleep(0.5)
    au=' and '.join(f"{a.get('given','')} {a.get('family','')}".strip() for a in m['author'])
    ti=re.sub(r'\s+',' ',m['title'][0]); ven=re.sub(r'\s+',' ',m['container-title'][0]); yr=(m.get('issued') or m['published'])['date-parts'][0][0]
    for k2 in ('published-print','published-online','issued'):
        if k2 in m: yr=m[k2]['date-parts'][0][0]; break
    vol=m.get('volume',''); iss=m.get('issue',''); pg=m.get('page','') .replace('-','--') if m.get('page') else ''
    typ='article' if m['type']=='journal-article' else 'inproceedings'
    f='journal' if typ=='article' else 'booktitle'
    e=f"@{typ}{{{k},\n  author={{{au}}},\n  title={{{{{ti}}}}},\n  {f}={{{ven}}},\n  year={{{yr}}},\n"
    if vol:e+=f"  volume={{{vol}}},\n"
    if iss:e+=f"  number={{{iss}}},\n"
    if pg:e+=f"  pages={{{pg}}},\n"
    e+=f"  doi={{{d}}}\n}}\n"; bib.append(e)
    chk.append(f"| {k} | {d} | {ti} | {au.replace(' and ',', ')} | {ven}, {yr}, vol {vol}, no {iss}, pp {pg} | verified in Crossref |")
bib.append("@inproceedings{mcmahan17,\n  author={H. Brendan McMahan and Eider Moore and Daniel Ramage and Seth Hampson and Blaise Ag{\\\"u}era y Arcas},\n  title={Communication-efficient learning of deep networks from decentralized data},\n  booktitle={Proc. Int. Conf. Artif. Intell. Statist. (AISTATS)},\n  year={2017},\n  note={arXiv:1602.05629}\n}\n")
chk.append("| mcmahan17 | arXiv:1602.05629 | Communication-Efficient Learning of Deep Networks from Decentralized Data | McMahan, Moore, Ramage, Hampson, Aguera y Arcas | AISTATS 2017 (PMLR 54) | the one arXiv-listed reference; verify on arxiv.org |")
open('../refs.bib','w').write('\n'.join(bib)); open('../refs_check.md','w').write('\n'.join(chk)+'\n')
