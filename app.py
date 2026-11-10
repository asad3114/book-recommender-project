from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from urllib.parse import quote
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- App Configuration & Styling ---
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "data" / "books.csv"


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def value_from(row: dict[str, str | None], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class Book:
    title: str
    author: str
    genre: str
    description: str

    @property
    def searchable_text(self) -> str:
        return " ".join(
            part for part in (
                f"title {self.title}" if self.title else "",
                f"author {self.author}" if self.author else "",
                f"genre {self.genre} {self.genre}" if self.genre else "",
                self.description,
            ) if part
        )


class BookRecommender:
    def __init__(self, books: list[Book]) -> None:
        if len(books) < 2:
            raise ValueError("The CSV must contain at least two valid books.")
        self.books = books
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(book.searchable_text for book in books)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(matrix)
        self.matrix = matrix

    @classmethod
    def from_csv(cls, path: Path) -> "BookRecommender":
        if not path.is_file():
            raise FileNotFoundError(f"Could not find the book CSV: {path}")

        books: list[Book] = []
        seen_titles: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("The CSV is empty or does not have a header row.")
            for raw_row in reader:
                row = {
                    normalize_column_name(key): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                title = value_from(row, "title", "book_title", "name")
                if not title or title.casefold() in seen_titles:
                    continue
                book = Book(
                    title=title,
                    author=value_from(row, "author", "book_author", "authors"),
                    genre=value_from(row, "genre", "genres", "category", "categories", "subject"),
                    description=value_from(
                        row, "description", "summary", "synopsis", "about", "abstract"
                    ),
                )
                books.append(book)
                seen_titles.add(title.casefold())

        if len(books) < 2:
            raise ValueError("The CSV needs at least two rows with a title column.")
        return cls(books)

    def recommend(self, book_index: int, count: int = 5) -> list[tuple[Book, float]]:
        if count < 1:
            return []
        neighbors = min(count + 1, len(self.books))
        distances, indices = self.model.kneighbors(
            self.matrix.getrow(book_index), n_neighbors=neighbors
        )
        recommendations: list[tuple[Book, float]] = []
        for distance, index in zip(distances[0], indices[0]):
            if index == book_index:
                continue
            similarity = max(0.0, 1.0 - float(distance)) * 100
            recommendations.append((self.books[index], similarity))
            if len(recommendations) == count:
                break
        return recommendations


# --- Helper Function to Fetch Book Cover via Open Library API ---
@st.cache_data(ttl=86400)
def get_book_cover_url(title: str, author: str) -> str | None:
    try:
        query = f"{title} {author}".strip()
        url = f"https://openlibrary.org/search.json?q={quote(query)}&limit=1"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            docs = data.get("docs", [])
            if docs:
                cover_i = docs[0].get("cover_i")
                if cover_i:
                    return f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg"
                isbns = docs[0].get("isbn")
                if isbns:
                    return f"https://covers.openlibrary.org/b/isbn/{isbns[0]}-M.jpg"
    except Exception:
        pass
    return None


# --- Load Model with Caching ---
@st.cache_resource
def load_recommender() -> BookRecommender:
    return BookRecommender.from_csv(DEFAULT_DATA)


try:
    recommender = load_recommender()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()


# --- Streamlit UI Layout ---
st.title("📚 Book Similarity Recommender")
st.markdown("Select or filter books below to find similar reads based on plot, genre, and author features.")

# --- Sidebar Controls & Filters ---
st.sidebar.header("Configuration")
top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)

st.sidebar.header("Filters")
all_genres = sorted(list(set(b.genre for b in recommender.books if b.genre)))
all_authors = sorted(list(set(b.author for b in recommender.books if b.author)))

selected_genres = st.sidebar.multiselect("Filter by Genre", all_genres)
selected_authors = st.sidebar.multiselect("Filter by Author", all_authors)

filtered_books = recommender.books
if selected_genres:
    filtered_books = [b for b in filtered_books if b.genre in selected_genres]
if selected_authors:
    filtered_books = [b for b in filtered_books if b.author in selected_authors]

book_titles = sorted([book.title for book in filtered_books])

if not book_titles:
    st.warning("No books match your selected filters. Try clearing or broadening your filters in the sidebar.")
    selected_title = None
else:
    selected_title = st.selectbox("Choose a book:", book_titles)

if selected_title:
    selected_index = next(i for i, b in enumerate(recommender.books) if b.title == selected_title)
    selected_book = recommender.books[selected_index]
    
    cover_url = get_book_cover_url(selected_book.title, selected_book.author)

    st.markdown("### Selected Book")
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            if cover_url:
                st.image(cover_url, width=120)
            else:
                st.image("https://via.placeholder.com/120x180?text=No+Cover", width=120)
        with col2:
            st.info(f"**{selected_book.title}**" + (f" by *{selected_book.author}*" if selected_book.author else ""))
            if selected_book.genre:
                st.caption(f"**Genre:** {selected_book.genre}")
            if selected_book.description:
                st.write(selected_book.description)

        # External Reading Links for Selected Book
        sel_query = quote(f"{selected_book.title} {selected_book.author}")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.link_button("📚 View on Goodreads", f"https://www.goodreads.com/search?q={sel_query}", use_container_width=True)
        with b_col2:
            st.link_button("📖 Read/Search Open Library", f"https://openlibrary.org/search?q={sel_query}", use_container_width=True)

    st.divider()

    # Get and display recommendations with covers & interactive read buttons
    st.markdown(f"### Recommended Reads (Top {top_n})")
    recommendations = recommender.recommend(selected_index, count=top_n)

    if not recommendations:
        st.warning("No recommendations available.")
    else:
        for idx, (rec_book, similarity) in enumerate(recommendations, 1):
            rec_cover_url = get_book_cover_url(rec_book.title, rec_book.author)
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if rec_cover_url:
                        st.image(rec_cover_url, width=90)
                    else:
                        st.image("https://via.placeholder.com/90x130?text=No+Cover", width=90)
                with col2:
                    author_text = f" by *{rec_book.author}*" if rec_book.author else ""
                    genre_text = f" — *{rec_book.genre}*" if rec_book.genre else ""
                    st.markdown(f"**{idx}. {rec_book.title}**{author_text}{genre_text}")
                    if rec_book.description:
                        st.write(rec_book.description)
                with col3:
                    st.metric(label="Match", value=f"{similarity:.1f}%")
                
                # Recommendation action buttons
                rec_query = quote(f"{rec_book.title} {rec_book.author}")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.link_button("📚 Goodreads", f"https://www.goodreads.com/search?q={rec_query}", use_container_width=True)
                with r_col2:
                    st.link_button("📖 Open Library", f"https://openlibrary.org/search?q={rec_query}", use_container_width=True)

                st.markdown("---")