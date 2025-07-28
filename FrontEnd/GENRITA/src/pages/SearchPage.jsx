// --- FILE: src/pages/SearchPage.js ---
import { useState, useEffect } from 'react';
import { mockApi } from '../data/mockApi';
import BookCard from '../components/common/BookCard';
const SearchPage = ({ setPage, setSelectedBookId }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    useEffect(() => {
        const handler = setTimeout(() => {
            if (query) {
                setIsLoading(true);
                mockApi.searchBooks(query).then(data => {
                    setResults(data);
                    setIsLoading(false);
                });
            } else {
                setResults([]);
            }
        }, 300);
        return () => clearTimeout(handler);
    }, [query]);
    const viewDetails = (bookId) => { setSelectedBookId(bookId); setPage('bookDetails'); };
    return (
        <div className="p-4 md:p-6">
            <input type="text" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search by title or author..." className="w-full px-4 py-2 mb-6 bg-[var(--color-background-secondary)] border border-[var(--color-border)] rounded-lg" />
            {isLoading ? <p className="text-center text-[var(--color-text-muted)]">Searching...</p> : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
                    {results.map(b => <BookCard key={b.id} book={b} onClick={() => viewDetails(b.id)} />)}
                </div>
            )}
        </div>
    );
}
export default SearchPage