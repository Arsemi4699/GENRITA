// --- FILE: src/pages/LibraryPage.jsx (REVISED) ---
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import BookCard from '../components/common/BookCard';
import { api } from '../services/api'; // Changed from mockBooks

const LibraryPage = ({ setPage, setSelectedBookId }) => {
    const { user } = useAuth();
    const [libraryBooks, setLibraryBooks] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // The logic is now completely rewritten to fetch real book data
        if (user?.libraryBookIds && user.libraryBookIds.length > 0) {
            setIsLoading(true);
            Promise.all(user.libraryBookIds.map(id => api.fetchBookDetails(id)))
                .then(books => {
                    setLibraryBooks(books.filter(Boolean)); // Filter out any potential nulls if a book fetch fails
                    setIsLoading(false);
                })
                .catch(err => {
                    console.error("Failed to load library books", err);
                    setIsLoading(false);
                });
        } else {
            setLibraryBooks([]);
            setIsLoading(false);
        }
    }, [user]);

    const viewDetails = (bookId) => { setSelectedBookId(bookId); setPage('bookDetails'); };

    if (isLoading) return <div className="p-4 md:p-6 text-center">Loading Library...</div>;

    return (
        <div className="p-4 md:p-6">
            {libraryBooks.length === 0 ? (
                <p className="text-center text-[var(--color-text-muted)] mt-8">Your library is empty.</p>
            ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
                    {libraryBooks.map(b => b && <BookCard key={b.id} book={b} onClick={() => viewDetails(b.id)} />)}
                </div>
            )}
        </div>
    );
};
export default LibraryPage;
