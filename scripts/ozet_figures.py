# Simple summary figures for a non-expert reader (Turkish labels); writes reports/ozet/*.png.
# Usage: python3 scripts/ozet_figures.py   (needs reports/pmnist_fullstream/curves.csv and reports/pmnist_widenet/curves.csv)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="reports/ozet"
COL={"baseline":"#2a78d6","sin":"#eb6834","tanh":"#1baf7a","tri":"#eda100","weight_clipping":"#e87ba4"}
NAME={"baseline":"Standart ağ","sin":"Sinüs sınırı","tanh":"tanh sınırı","tri":"Üçgen dalga (yansıma)","weight_clipping":"Weight Clipping (kırpma)"}
STY={"weight_clipping":(0,(4,2))}
plt.rcParams.update({"font.size":11,"axes.titlesize":12,"axes.labelsize":11,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
    "xtick.color":"#52514e","ytick.color":"#52514e","axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb"})
def style(ax):
    ax.grid(True,color="#e6e5e1",linewidth=0.8); ax.set_axisbelow(True)
def curves(path, metric, labels, win):
    d=pd.read_csv(path); out={}
    for lab in labels:
        s=d[d.label==lab].dropna(subset=[metric]).groupby("task")[metric].mean().sort_index()
        out[lab]=s.rolling(win,min_periods=1,center=True).mean()
    return out
def label_end(ax, series, dy=None, fs=10):
    ends=sorted([(s.iloc[-1],lab) for lab,s in series.items()])
    ys=[e[0] for e in ends]; lo,hi=ax.get_ylim(); mind=(hi-lo)*0.045
    for i in range(1,len(ys)):
        if ys[i]-ys[i-1]<mind: ys[i]=ys[i-1]+mind
    for (v,lab),y in zip(ends,ys):
        s=series[lab]; ax.annotate(NAME[lab],(s.index[-1],v),xytext=(s.index[-1]*1.015,y),fontsize=fs,color="#0b0b0b",va="center",
                    arrowprops=dict(arrowstyle="-",color="#b8b7b2",lw=0.8) if abs(y-v)>1e-9 else None)
# ---------- F1: 800 görev ----------
labs=["baseline","sin","tanh","tri","weight_clipping"]
c=curves("reports/pmnist_fullstream/curves.csv","online_accuracy",labs,10)
fig,ax=plt.subplots(figsize=(10,5.2)); style(ax)
for lab in labs: ax.plot(c[lab].index,c[lab].values,color=COL[lab],lw=2,ls=STY.get(lab,"-"),label=NAME[lab])
ax.set_xlim(0,800*1.3); ax.set_ylim(0.68,0.96); label_end(ax,c)
ax.set_xlabel("Görev sırası (her görev yeni bir piksel karışımı, 60.000 örnek)"); ax.set_ylabel("Görevdeki doğruluk")
ax.set_title("Uzun akış (800 görev, 3×100 ağ): plastisite kaybı var mı?",loc="left",fontweight="bold",pad=22)
ax.set_xticks(range(0,801,100))
ax.text(0,1.015,"3 tohum ortalaması, 10 görevlik kayan ortalama. Yükseklik = ağın yeni görevi ne kadar iyi öğrendiği.",transform=ax.transAxes,fontsize=9.5,color="#52514e")
ax.legend(loc="lower left",frameon=False,fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/1_uzun_akis_800_gorev.png",dpi=150); plt.close(fig)
# ---------- F2: üç harita ----------
fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
th=np.linspace(-2.4,2.4,600)
panels=[("sin","W = A·sin(Θ)  — yumuşak sınır",np.clip(np.sin(th),-1,1),"Sınıra yaklaşınca eğim 0'a iner:\nadım küçülür, parametre donar"),
        ("tri","W = A·tri(Θ)  — yansımalı sınır",2/np.pi*np.arcsin(np.sin(th)),"Eğim her yerde ±1:\nsınıra çarpınca geri yansır, adım küçülmez"),
        ("weight_clipping","Weight Clipping — kırpma",np.clip(th,-1,1),"Sınırı aşan değer sınıra çekilir;\nadım küçülmez")]
for ax,(lab,title,y,note) in zip(axs,panels):
    style(ax); ax.axhline(1,color="#b8b7b2",ls="--",lw=1); ax.axhline(-1,color="#b8b7b2",ls="--",lw=1)
    ax.plot(th,y,color=COL[lab],lw=2.4); ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
    ax.set_xlabel("serbest parametre Θ"); ax.text(0.03,0.08,note,transform=ax.transAxes,fontsize=9.5,color="#0b0b0b",va="bottom")
    ax.text(2.3,1.05,"+A",color="#52514e",fontsize=9,ha="right"); ax.set_ylim(-1.35,1.35)
axs[0].set_ylabel("ağın kullandığı ağırlık W / A")
fig.suptitle("Üç sınır türü: ağırlık hiçbir zaman ±A'yı aşamaz, fark sınırda ne olduğunda",x=0.01,ha="left",fontsize=12,fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(f"{OUT}/2_uc_sinir_turu.png",dpi=150); plt.close(fig)
# ---------- F3: mekanizma ----------
fig,axs=plt.subplots(1,3,figsize=(13,4.3))
sat=curves("reports/pmnist_fullstream/curves.csv","sat_frac/all",["sin","tanh","tri"],1)
jac=curves("reports/pmnist_fullstream/curves.csv","jac_sq_mean/all",["sin","tanh","tri"],1)
wab=curves("reports/pmnist_fullstream/curves.csv","w_abs_mean/all",labs,1)
for ax,(data,title,ylab) in zip(axs,[(sat,"Donmuş parametre payı","sınırda donan parametrelerin oranı"),
                                     (jac,"Etkin adım çarpanı (∂W/∂Θ)²","1 = tam adım, 0 = hiç hareket yok"),
                                     (wab,"Ağırlıkların ortalama büyüklüğü |w|","ortalama |w|")]):
    style(ax)
    for lab,s in data.items(): ax.plot(s.index,s.values,color=COL[lab],lw=2,ls=STY.get(lab,"-"))
    ax.set_xlim(0,800*1.55); ax.set_xticks(range(0,801,200)); ax.set_title(title,loc="left",fontsize=11,fontweight="bold"); ax.set_xlabel("Görev"); ax.set_ylabel(ylab)
    lo,hi=ax.get_ylim(); ax.set_ylim(lo,hi); label_end(ax,data,fs=9)
fig.suptitle("Neden sinüs kaybediyor, üçgen dalga kazanıyor? (800 görev, 3 tohum ortalaması)",x=0.01,ha="left",fontsize=12,fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(f"{OUT}/3_mekanizma.png",dpi=150); plt.close(fig)
# ---------- F4: üç protokol ----------
prot=[("Küçük ağ (3×100)\n200 görev × 5.000 örnek, 10 tohum",{"baseline":0.743,"sin":0.760,"tanh":0.761,"tri":0.839,"weight_clipping":0.839}),
      ("Küçük ağ (3×100)\n800 görev × 60.000 örnek, 3 tohum",{"baseline":0.835,"sin":0.716,"tanh":0.828,"tri":0.939,"weight_clipping":0.940}),
      ("Geniş ağ (3×2000)\n200 görev × 5.000 örnek, 3 tohum",{"baseline":0.863,"sin":0.864,"tanh":0.864,"tri":0.862})]
fig,axs=plt.subplots(1,3,figsize=(13,4.6))
for ax,(title,vals) in zip(axs,prot):
    style(ax); ax.grid(axis="x",visible=False)
    ks=list(vals); xs=np.arange(len(ks))
    ax.bar(xs,[vals[k] for k in ks],color=[COL[k] for k in ks],width=0.62)
    ax.axhline(vals["baseline"],color="#52514e",ls=(0,(3,2)),lw=1)
    for x,k in zip(xs,ks): ax.text(x,vals[k]+0.006,f"{vals[k]:.3f}",ha="center",fontsize=9,color="#0b0b0b")
    SHORT={"baseline":"Standart\nağ","sin":"Sinüs\nsınırı","tanh":"tanh\nsınırı","tri":"Üçgen\ndalga","weight_clipping":"Weight\nClipping"}
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[k] for k in ks],fontsize=9)
    ax.set_ylim(0.65,0.97); ax.set_title(title,loc="left",fontsize=10.5,fontweight="bold")
axs[0].set_ylabel("Son görevlerdeki doğruluk (son pencere)")
fig.suptitle("Akışın sonunda ağ yeni görevleri ne kadar iyi öğreniyor? Üç deney ölçeği",x=0.01,ha="left",fontsize=12,fontweight="bold")
fig.text(0.01,0.01,"Kesikli çizgi: standart ağın değeri (referans). Yükseklik = akışın son görevlerindeki ortalama doğruluk; yüksek = plastisite korunmuş.",fontsize=9,color="#52514e")
fig.tight_layout(rect=(0,0.04,1,0.94)); fig.savefig(f"{OUT}/4_uc_olcek_son_dogruluk.png",dpi=150); plt.close(fig)
# ---------- F5: geniş ağ ----------
labs4=["baseline","sin","tanh","tri"]
c=curves("reports/pmnist_widenet/curves.csv","online_accuracy",labs4,5)
fig,ax=plt.subplots(figsize=(10,5)); style(ax)
for lab in labs4: ax.plot(c[lab].index,c[lab].values,color=COL[lab],lw=2,label=NAME[lab])
ax.set_xlim(0,200*1.3); ax.set_ylim(0.835,0.88); label_end(ax,c)
ax.set_xlabel("Görev sırası (200 görev × 5.000 örnek)"); ax.set_ylabel("Görevdeki doğruluk")
ax.set_title("Geniş ağ (3×2000): kimse plastisite kaybetmiyor, sınır fark yaratmıyor",loc="left",fontweight="bold",pad=22)
ax.set_xticks(range(0,201,25))
ax.text(0,1.015,"3 tohum ortalaması, 5 görevlik kayan ortalama. Dikey eksen dar: tüm eğriler 0.84–0.88 bandında.",transform=ax.transAxes,fontsize=9.5,color="#52514e")
ax.legend(loc="lower right",frameon=False,fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{OUT}/5_genis_ag_200_gorev.png",dpi=150); plt.close(fig)
print("done")
