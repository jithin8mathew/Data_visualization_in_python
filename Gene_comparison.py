import pandas as pd
import re

# data = open('comparison.txt','r+').read()
def comparison(file_name):
	data = open(file_name,'r+').read()

	data = re.findall(r'(\d{1,}\s[\+|\-]\s{1,}\d\s[A-z]{1,}\s{1,}\d{1,}\s\-\s{1,}\d{1,}\s{1,}[\-|\d]{1,}\.\d{1,}\s{1,}\d{1,}\s{1,}\-\s{1,}\d{1,}\s{1,}\d{1,})',data)
	data2=[]
	for x in data:
		tmp_lst=[]
		[tmp_lst.append(y) for y in x.split(' ') if y != '']
		data2.append(tmp_lst)

	data = pd.DataFrame(data2, index= [x for x in range(len(data))], columns=['G','Str','exon_no','Feature','Start','hyphen','End','Score','ORF_start','hyphen2','ORF_end','Len'])
	data=data.drop(columns=['hyphen','hyphen2'])

	# Uncomment the below line to generate csv file of exon sequence data
	#data.to_csv('Arabidopsis_exons.csv')

	data= data.groupby(['G'])
	print(data)
	lst=[]
	dct = {}
	for key, item in data:
		d = data.get_group(key)
		count = 0
		for i in d['G']:
			dct = {i:{d['exon_no'].iloc[count]:[d['Start'].iloc[count],d['End'].iloc[count],d['ORF_start'].iloc[count],d['ORF_end'].iloc[count],d['Len'].iloc[count]]}}
			count+=1
			lst.append(dct)
	return lst

o1=comparison('comparison.txt')
o2=comparison('comparison1.txt')

total_exon=[]
similar_exon_list=[]
similar_gene_list=[]
for x in o1:
	for y in o2:
		for key, values in x.items():
			for key1, values1 in y.items():
				if key == key1:
					if values.keys() == values1.keys():
						total_exon.append(list(values.values())[0])
						if list(values.values())[0] == list(values1.values())[0]:
							similar_exon_list.append(list(values.values())[0])
							similar_gene_list.append(str(key))
							#print(str(key))
						
#print(similar_exon_list)
#print(total_exon)
organism_1=[]
organism_2=[]
organism_1_gene_list=[]
organism_2_gene_list=[]

for x in o1:
	for key, values in x.items():
		organism_1.append(list(values.values())[0])
		organism_1_gene_list.append(str(key))
for y in o2:
	for key, values in y.items():
		organism_2.append(list(values.values())[0])
		organism_2_gene_list.append(str(key))

#print(gene_list)
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
venn3([set([str(x) for x in similar_exon_list]), set([str(z) for z in organism_1]), set([str(zz) for zz in organism_2])],set_labels = ('Exons in Glycine \n max and Arabidopsis', 'Glycine Max exons ' + str(len(organism_1)),'Arabidopsis exons '+str(len(organism_2))))
plt.title('Comparison of exons\n')
plt.show()

venn3([set([str(x) for x in similar_gene_list]), set([str(z) for z in organism_1_gene_list]), set([str(zz) for zz in organism_2_gene_list])],set_labels = ('Genes in Glycine \n max and Arabidopsis', 'Glycine Max Genes ' + str(len(set(organism_1_gene_list))),'Arabidopsis genes '+str(len(set(organism_2_gene_list)))))
plt.title('Comparison of genes\n')
plt.show()
