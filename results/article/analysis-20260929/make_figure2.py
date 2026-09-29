#!/usr/bin/env python3
"""Update the verdict panel from the verified numerical analysis.

Panels (a) and (b) are retained from the supplied manuscript. Panel (c) is
rendered as a separate statistical plot and assembled below them. No inference.
"""
from pathlib import Path
import csv, base64, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent

def main():
    rows={r['evidence_set']:r for r in csv.DictReader((ROOT/'verdict-analysis/verdict-summary.csv').open())}
    order=['jev','gemini','qwen','lfm26','jeff','laya_typed','lfm','include_all','include_none','majority_class']
    names=['Jev','Gemini','Qwen-4B','LFM-2.6B','Jeff','Laya typed','LFM-1.2B','All snippets','No snippets','Always Refuted (reference)']
    accuracy=np.array([float(rows[k]['accuracy_pct']) for k in order])
    low=np.array([float(rows[k]['accuracy_ci_low']) for k in order])
    high=np.array([float(rows[k]['accuracy_ci_high']) for k in order])
    y=np.arange(len(order))
    fig=plt.figure(figsize=(11.3,4.65),dpi=300)
    ax=fig.add_axes([.245,.15,.71,.72])
    ax.errorbar(accuracy[:9],y[:9],xerr=np.array([accuracy[:9]-low[:9],high[:9]-accuracy[:9]]),fmt='o',capsize=3,markersize=5,linewidth=1.2)
    ax.errorbar(accuracy[9],y[9],xerr=np.array([[accuracy[9]-low[9]],[high[9]-accuracy[9]]]),fmt='D',capsize=3,markersize=6,linewidth=1.2)
    ax.set_yticks(y,names,fontsize=12)
    ax.set_xlim(0,106);ax.set_ylim(9.65,-.65)
    ax.set_xticks([0,20,40,60,80,100]);ax.tick_params(axis='x',labelsize=11)
    ax.grid(axis='x',alpha=.25)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.set_xlabel('Mean accuracy, % (pointwise 95% claim-bootstrap interval)',fontsize=12,labelpad=10)
    ax.set_title('c) HerO verdict accuracy and fixed reference policy',loc='left',fontsize=15,fontweight='bold',pad=18)
    for yy,acc in zip(y,accuracy):ax.text(105,yy,f'{acc:.1f}',ha='right',va='center',fontsize=10)
    target=ROOT/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/'figure2c.png',dpi=300)
    with matplotlib.rc_context({'svg.fonttype':'none'}):fig.savefig(target/'figure2c.svg')
    plt.close(fig)
    ab=Image.open(target/'source_panels_ab.png').convert('RGB')
    c=Image.open(target/'figure2c.png').convert('RGB')
    if c.width!=ab.width:c=c.resize((ab.width,round(c.height*ab.width/c.width)),Image.Resampling.LANCZOS)
    combined=Image.new('RGB',(ab.width,ab.height+c.height),(255,255,255));combined.paste(ab,(0,0));combined.paste(c,(0,ab.height))
    combined.save(target/'figure2.png',dpi=(300,300))
    # SVG retains the original raster panels and a vector-rendered new panel.
    svg=(target/'figure2c.svg').read_text();svg=svg[svg.index('<svg'):]
    vb=re.search(r'viewBox="([^"]+)"',svg).group(1).split();vw,vh=float(vb[2]),float(vb[3]);scale=ab.width/vw
    svg=re.sub(r'<svg[^>]*>',f'<svg x="0" y="{ab.height}" width="{ab.width}" height="{c.height}" viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg">',svg,count=1)
    b64=base64.b64encode((target/'source_panels_ab.png').read_bytes()).decode()
    outer=f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{ab.width}" height="{combined.height}" viewBox="0 0 {ab.width} {combined.height}"><image x="0" y="0" width="{ab.width}" height="{ab.height}" xlink:href="data:image/png;base64,{b64}"/>{svg}</svg>'
    (target/'figure2.svg').write_text(outer)
    print('Updated Figure 2:',combined.size)

if __name__=='__main__':main()
