import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
from scipy.stats import mannwhitneyu
from skbio.stats import ordination
from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity
from deicode.preprocessing import rclr
from Bio import Phylo
from skbio.tree import TreeNode
from skbio import read
import random
from ete3 import Tree

folder = '/Users/robynwright/Dropbox/Langille_Lab_postdoc/OED_project/analysis/'
colors = {'progression':'#CD5C5C', 'no progression':'#2980B9'}
colors_time_groups = {'NP':'#2980B9', '<1':'#A93226', '1<2':'#E74C3C', '2<3':'#DC7633', '3<4':'#FF7043', '4<6':'#F4D03F', '>6':'#AFB42B', 'Other':'k'}

ft_relabun_fn, ft_rare_fn, ft_rclr_fn, genus_relabun_fn, genus_rare_fn, genus_rclr_fn = folder+'processing/ft_relabun.csv', folder+'processing/ft_rare.csv', folder+'processing/ft_rclr.csv', folder+'processing/genus_relabun.csv', folder+'processing/genus_rare.csv', folder+'processing/genus_rclr.csv'
tree_fn, genus_tree_fn = folder+'qiime2/exports/tree.nwk', folder+'processing/genus_tree.nwk'
agglom_relabun_ft, agglom_rclr_ft = folder+'processing/agglom_relabun.csv', folder+'processing/agglom_rclr.csv'
agglom_tree_fn = folder+'processing/agglom_tree.tree'

sig_hits = {}

with open(folder+'processing/matches.dict', 'rb') as f:
    matches_dict = pickle.load(f)

this_df = pd.read_csv(folder+'differential_abundance/maaslin_EC_progressors_specific_full_model/significant_results.tsv', index_col=0, header=0, sep='\t')
rename_ec = {}
for row in this_df.index:
  rename_ec[row] = row.replace('EC.', 'EC:')
this_df = this_df.rename(index=rename_ec)
for row in this_df.index:
  if this_df.loc[row, 'qval'] > 0.2: continue
  elif this_df.loc[row, 'qval'] > 0.1 and this_df.loc[row, 'coef'] < 0.0001: continue
  hit = [this_df.loc[row, 'value'],this_df.loc[row, 'coef'], this_df.loc[row, 'qval']]
  sig_hits[row] = hit
  
info = pd.read_csv(folder+'PICRUST/EC_metagenome_out/pred_metagenome_unstrat_descrip.tsv', sep='\t', index_col=0, header=0)
    
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
md = pd.read_csv(folder+'OM_metadata_groups.csv', index_col=0, header=0)
md.index = md.index.map(str)
ft = pd.read_csv(folder+'processing/pred_metagenome_unstrat_relabun.tsv', index_col=0, header=0)
x = {'<1':0, '1<2':3, '2<3':6, '3<4':9, '4<6':12, '>6':15}

sig = [s for s in sig_hits]
plt.figure(figsize=(20,20))
#locs = [[], [0,0,2,2,4,4,6,6,8,8]]
for t in range(len(sig)):
  ax = plt.subplot2grid((10,4),(t, 1))
  plt.sca(ax)
  vals_x, vals_y = [], []
  for sample in ft.columns:
    group = md.loc[sample, 'Time_to_progression_grouped']
    years = md.loc[sample, 'Total_Number_of_Months_Followed_or_to_Progression']/12
    if group == 'NP': continue
    val = ft.loc[sig[t], sample]
    scat = plt.scatter(years, val, color=colors['progression'], alpha=0.5)
    vals_x.append(years), vals_y.append(val)
  theta = np.polyfit(vals_x, vals_y, 1)
  y_line = theta[1] + theta[0] * np.array(vals_x)
  li = plt.plot(vals_x, y_line, 'k-')
  string = 'Coefficient='+str(round(sig_hits[sig[t]][1], 5))+', $q$='+str(round(sig_hits[sig[t]][2], 3))
  tx = plt.text(0.01, 0.95, string, transform=ax.transAxes, fontsize=14, va='top', ha='left', bbox=props)
  #ti = plt.title(sig[t]+': '+info.loc[sig[t], 'description'], fontweight='bold')
  yl = plt.title('Relative abundance', fontweight='bold')
  yl = plt.ylabel('Relative abundance (%)')
  xl = plt.xlabel('Years to progression')
  ax = plt.subplot2grid((10,4),(t, 0))
  plt.sca(ax)
  vals_x, vals_y = [], []
  groups = {}
  for sample in ft.columns:
    group = md.loc[sample, 'Time_to_progression_grouped']
    years = md.loc[sample, 'Total_Number_of_Months_Followed_or_to_Progression']/12
    if group == 'NP': continue
    val = ft.loc[sig[t], sample]
    val_con = []
    x_P, x_NP = np.random.normal(x[group], 0.1, 1), np.random.normal(x[group]+1, 0.1, 1)
    for m in matches_dict[str(sample)]:
      vcon = ft.loc[sig[t], m]
      val_con.append(vcon)
      line = plt.plot([x_P, x_NP], [val, vcon], 'k-', alpha=0.05)
      scat = plt.scatter(x_NP, vcon, color=colors['no progression'], alpha=0.5)
    scat = plt.scatter(x_P, val, color=colors['progression'], alpha=0.5)
    if group in groups: groups[group] = [groups[group][0]+[val], groups[group][1]+val_con]
    else: groups[group] = [[val], val_con]
  for group in groups:
    box = ax.boxplot(groups[group], positions=[x[group], x[group]+1], widths=0.8, showfliers=False)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']: plt.setp(box[item], color='k')
  yl = plt.ylabel('Diversity')
  #ti = plt.ylabel(sig[t], fontweight='bold')
  xl = plt.xticks([x[val]+0.5 for val in x],[val for val in x])
  xl = plt.xlabel('Years to progression (grouped)')
  yl = plt.ylabel('Relative abundance (%)')
  ti = plt.title('Relative abundance (%)', fontweight='bold')
  te = plt.text(-0.3, 0.5, sig[t]+'\n'+info.loc[sig[t], 'description'].replace(' ', '\n'), fontweight='bold', transform=ax.transAxes, rotation=90, ha='center', va='center')

# ft = pd.read_csv(folder+'processing/pred_metagenome_unstrat_relabun.tsv', index_col=0, header=0)
new_ft = []
div = 'faith_pd'
title_name = "Faith's phylogenetic diversity"
for hit in sig:
  nhit = hit.replace(':', '.')
  this_alpha = pd.read_csv(folder+'diversity/functional_diversity/alpha_diversity_'+nhit+'_genus.csv', index_col=0, header=0)
  this_alpha = this_alpha.loc[:, [div]]
  this_alpha = this_alpha.rename(columns={div:hit})
  new_ft.append(this_alpha)

new_ft = pd.concat(new_ft).fillna(value=0)
new_ft = new_ft.groupby(by=new_ft.index, axis=0).sum().transpose()
ft = new_ft
ft.columns = ft.columns.map(str)

#locs[1] = [1,1,3,3,5,5,7,7,9,9]
for t in range(len(sig)):
  ax = plt.subplot2grid((10,4),(t, 3))
  plt.sca(ax)
  vals_x, vals_y = [], []
  for sample in ft.columns:
    group = md.loc[sample, 'Time_to_progression_grouped']
    years = md.loc[sample, 'Total_Number_of_Months_Followed_or_to_Progression']/12
    if group == 'NP': continue
    val = ft.loc[sig[t], sample]
    scat = plt.scatter(years, val, color=colors['progression'], alpha=0.5)
    vals_x.append(years), vals_y.append(val)
  theta = np.polyfit(vals_x, vals_y, 1)
  y_line = theta[1] + theta[0] * np.array(vals_x)
  li = plt.plot(vals_x, y_line, 'k-')
  yl = plt.title(title_name, fontweight='bold')
  yl = plt.ylabel('Diversity')
  xl = plt.xlabel('Years to progression')
  #plt.yscale('symlog')
  ax = plt.subplot2grid((10,4),(t, 2))
  plt.sca(ax)
  vals_x, vals_y = [], []
  groups = {}
  for sample in ft.columns:
    group = md.loc[sample, 'Time_to_progression_grouped']
    years = md.loc[sample, 'Total_Number_of_Months_Followed_or_to_Progression']/12
    if group == 'NP': continue
    val = ft.loc[sig[t], sample]
    val_con = []
    x_P, x_NP = np.random.normal(x[group], 0.1, 1), np.random.normal(x[group]+1, 0.1, 1)
    for m in matches_dict[str(sample)]:
      vcon = ft.loc[sig[t], m]
      val_con.append(vcon)
      line = plt.plot([x_P, x_NP], [val, vcon], 'k-', alpha=0.05)
      scat = plt.scatter(x_NP, vcon, color=colors['no progression'], alpha=0.5)
    scat = plt.scatter(x_P, val, color=colors['progression'], alpha=0.5)
    if group in groups: groups[group] = [groups[group][0]+[val], groups[group][1]+val_con]
    else: groups[group] = [[val], val_con]
  for group in groups:
    box = ax.boxplot(groups[group], positions=[x[group], x[group]+1], widths=0.8, showfliers=False)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']: plt.setp(box[item], color='k')
  ti = plt.title(title_name, fontweight='bold')
  yl = plt.ylabel('Diversity')
  xl = plt.xlabel('Years to progression (grouped)')
  #ti = plt.ylabel(sig[t], fontweight='bold')
  xl = plt.xticks([x[val]+0.5 for val in x],[val for val in x])
  #if locs[0][t] == 1: yl = plt.title(title_name)


plt.tight_layout()
plt.savefig(folder+'figures/FigureS4_div_'+div+'_flipped_split.png', dpi=600, bbox_inches='tight')
