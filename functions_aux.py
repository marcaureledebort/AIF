
def reco_imdb (metadata,genre=-1,year=-1):
    subset=metadata.copy()
    if year!='Toutes les années':
        subset=subset[subset.year==int(year)]
    if genre !="Tous les genres":
        subset=subset[subset[genre]==1]
    best_movies = subset.sort_values('imdb_score', ascending=False)
    return best_movies[['title', 'imdb_score']].head(10)

def reco_plot(title, sim_matrix, indices, titles):
    idx = indices[title]
    if idx is None:
        print(f"Title '{title}' not found in indices.")
        return None
    else:
        recos = sim_matrix[idx].argsort()[1:6]
        recos_titles = titles.iloc[recos]
        print(recos_titles)
        return recos_titles
