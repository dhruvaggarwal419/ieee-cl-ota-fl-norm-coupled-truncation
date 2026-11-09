import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams['font.family']='serif'; plt.rcParams['font.size']=7.5
fig,ax=plt.subplots(figsize=(7.1,2.75)); ax.set_xlim(0,100); ax.set_ylim(-7,62); ax.axis('off')
C={'dev':'#DAE8FC','rule':'#FFE6CC','mac':'#E1D5E7','rx':'#D5E8D4','grp':'#F7F7F7'}
def box(x,y,w,h,t,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.2,rounding_size=1.0',fc=C[c],ec='#555',lw=0.7))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',linespacing=1.25)
def arrow(x1,x2,y):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='-|>',lw=0.9,color='#333',shrinkA=0,shrinkB=0))
rows=[(34,'Baseline: common gain $\\eta$ (norm-coupled)','Common gain $\\eta$\nsend if $|h_k|^2\\geq\\eta^2G_k^2/P$','Server\n$\\hat g=y/(K\\eta)$',
  'Silence depends on $G_k$: devices with large gradients are silent more often.\nExpected update = gradient of a negative kernel density (mean shift).','#A00000'),
 (4,'Proposed: norm-aware gain $a_k$','Gain $a_k$ from Lambert W\nso that $a_k p_k=\\beta$','Server\n$\\hat g=y/(K\\beta)$',
  'Every device has the same expected received gain $\\beta$: unbiased.\n$\\beta=\\rho$ times the smallest ceiling $u_k$, and $u_k\\propto 1/G_k$. Noise scales with the largest norm.','#106010')]
for y0,title,rule,srv,note,nc in rows:
    ax.add_patch(Rectangle((0.5,y0-3.5),99,27,fc=C['grp'],ec='#999',lw=0.6,ls='--'))
    ax.text(2,y0+20.3,title,va='center',fontsize=8)
    box(3,y0+6.5,18,9,'Device $k$\ngradient $g_k$, norm $G_k$','dev')
    box(31,y0+6.5,27,9,rule,'rule')
    box(66,y0+6.5,16,9,'Channel adds the\nsignals and noise','mac')
    box(88,y0+6.5,10,9,srv,'rx')
    arrow(21.4,30.6,y0+11); arrow(58.4,65.6,y0+11); arrow(82.4,87.6,y0+11)
    ax.text(50,y0+0.3,note,ha='center',va='center',color=nc,fontsize=7,linespacing=1.3)
# legend
for i,(k,l) in enumerate([('dev','Device'),('rule','Gain rule'),('mac','Channel and noise'),('rx','Server')]):
    x=12+i*22; ax.add_patch(Rectangle((x,-5.2),2.2,2.2,fc=C[k],ec='#555',lw=0.6)); ax.text(x+3,-4.1,l,va='center',fontsize=7)
fig.savefig('../fig_overview.pdf',bbox_inches='tight',pad_inches=0.02)
