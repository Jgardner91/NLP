import pandas as pd 
import streamlit as st 
import altair as alt
from PIL import Image 
import pickle 

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")




def get_titles():
	WINE = pd.read_csv('WINENLP.csv')
	WINE.drop_duplicates(subset=['description'],inplace=True)
	WINE.reset_index(inplace=True,drop=True)
	length = WINE.shape
	titles = []
	for i in range(20000):
		titles.append(WINE['title'][i])

	return titles ,WINE


def stream_lit(titles):

	# loading in data 
	WINES = pd.read_pickle("WINESINFO.pkl")
	COS_MATRIX = pd.read_pickle("COSMAT.pkl")
	ATTR = pd.read_pickle("attributes.pkl")
	cos_df = pd.read_pickle("cosine_mat.pkl")
	
	# making wine map 
	map_wines = {WINES['title'][i]:i for i in range(20000)}
	
	# making select box 
	drop_down = st.sidebar.selectbox("Features",['Home Page','Recomendation System','Price Constraint','EDA'])
	if drop_down == 'Recomendation System':
		image = Image.open('sommolier.jpeg')
		st.image(image)

		st.sidebar.write(" ")
		st.sidebar.write("This reccomendation system helps you find wines similar to your favorite wine")
		st.sidebar.write("Search through catelogue by typing specific wine name or sytle")
		st.sidebar.write("Example Search ---> 'Pinot Noir' or 'Heron 2013 Pinot Noir'")



		wine_selection = st.selectbox("Select Wine",
			titles)

		# making similarity list 
		MAKE_REC = [[COS_MATRIX[map_wines[wine_selection]][i],i] for i in range(20000)]
		# sorting from most similar to least similar 
		SORTED_REC = sorted(MAKE_REC,reverse=True)
	
		for i in range(30):
			if i == 0:
				st.write("WINE OF COMPARISON")
				st.write("")
				st.write("Name: ",WINES['title'][SORTED_REC[i][1]])
				st.write("Variety: ",WINES['variety'][SORTED_REC[i][1]])
				st.write("Price: ", WINES['price'][SORTED_REC[i][1]])
				st.write("Description: ",WINES['description'][SORTED_REC[i][1]])
				st.write("TOP 20 MOST SIMILAR")
		
			else:
				st.write("Name: ",WINES['title'][SORTED_REC[i][1]])
				st.write("Variety: ",WINES['variety'][SORTED_REC[i][1]])
				st.write("Description: ",WINES['description'][SORTED_REC[i][1]])
				st.write("Price: ", WINES['price'][SORTED_REC[i][1]])
				st.write("")
	
	if drop_down == 'Home Page':
		st.title("Analyzing Wine Style and Structure")
		st.markdown("          *Using NLP*") 
		st.markdown("---")
		
		image = Image.open('app.jpeg')
		st.image(image)

	if drop_down == 'EDA':
		img = Image.open('winemap.jpeg')
		st.image(image)

	if drop_down == 'Price Constraint':

		img = Image.open('wine_price.png')
		st.image(img)
		wine_choice = st.selectbox("Select Wine",
			titles)
		min_price = st.number_input("Minimum Price",20)
		max_price = st.number_input("Maximum Price",40)
		

		df_similarity = pd.DataFrame(cos_df[map_wines[wine_choice]])
		ATTR['Similarity'] = df_similarity
		Sorted = ATTR.sort_values(by='Similarity',ascending=False)
		Sorted.reset_index(inplace=True,drop=True)
		mask = (Sorted['price'] >= min_price) & (Sorted['price'] <= max_price)
		Price_constraint = Sorted[mask]
		Price_constraint.reset_index(inplace=True,drop=True)

		for i in range(30):
			if i == 0:
				st.write("WINE OF COMPARISON")
				st.write("")
				st.write("Name: ",Price_constraint['title'][i])
				st.write("Variety: ",Price_constraint['variety'][i])
				st.write("Price: ", Price_constraint['price'][i])
				st.write("Description: ",Price_constraint['description'][i])
				st.write("TOP 20 MOST SIMILAR")

			if i > 0:
				st.write("")
				st.write("Name: ",Price_constraint['title'][i])
				st.write("Variety: ",Price_constraint['variety'][i])
				st.write("Price: ", Price_constraint['price'][i])
				st.write("Description: ",Price_constraint['description'][i])
				







		


	

def main():
	
	
	titles = get_titles()
	stream_lit(titles[0])
	
	

if __name__ == '__main__':
	main()