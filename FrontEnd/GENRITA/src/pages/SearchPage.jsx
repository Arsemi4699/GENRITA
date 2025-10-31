// --- FILE: src/pages/SearchPage.jsx (REVISED) ---
import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import BookCard from '../components/common/BookCard';
import SearchFilters from '../components/common/SearchFilters';

const SearchPage = ({ setPage, setSelectedBookId }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    // Expanded state to include all filter options
    const [filters, setFilters] = useState({
        order: 0,
        searchMode: 0,
        publishedDate: '', // Default to empty
    });

    const performSearch = useCallback(() => {
        if (query.trim()) {
            setIsLoading(true);
            api.searchBooks(query, filters)
                .then(data => {
                    setResults(data);
                    console.log(query);
                    console.log(filters);
                    console.log(data);
                    setIsLoading(false);
                })
                .catch(err => {
                    console.error("Search failed:", err);
                    setResults([]);
                    setIsLoading(false);
                });
        } else {
            setResults([]);
        }
    }, [query, filters]);

    useEffect(() => {
        const handler = setTimeout(() => {
            performSearch();
        }, 500); // Increased delay slightly for better UX with more filters

        return () => {
            clearTimeout(handler);
        };
    }, [performSearch]);

    const viewDetails = (bookId) => {
        setSelectedBookId(bookId);
        setPage('bookDetails');
    };

    return (
        <div className="p-4 md:p-6 max-w-7xl mx-auto">
            <input
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search by title or author..."
                className="w-full px-4 py-2 mb-4 bg-[var(--color-background-secondary)] border border-[var(--color-border)] rounded-lg"
            />

            <SearchFilters filters={filters} onFilterChange={setFilters} />

            {isLoading ? (
                <p className="text-center text-[var(--color-text-muted)]">Searching...</p>
            ) : (
                query && results.length === 0 ? (
                    <p className="text-center text-[var(--color-text-muted)] mt-8">No results found for "{query}".</p>
                ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
                        {results.map(b => <BookCard key={b.id} book={b} onClick={() => viewDetails(b.id)} />)}
                    </div>
                )
            )}
        </div>
    );
}
export default SearchPage;
