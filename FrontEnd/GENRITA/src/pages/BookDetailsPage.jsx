// --- FILE: src/pages/BookDetailsPage.js ---
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useBookStatus } from '../hooks/useBookStatus';
import { BookDetailsSkeleton } from '../components/common/SkeletonLoader';
import { mockApi } from '../data/mockApi';
import { useAudioSettings } from '../contexts/AudioSettingsContext'

const BookDetailsPage = ({ bookId, setPage }) => {
    const [book, setBook] = useState(null);
    const { user, addToLibrary } = useAuth();
    const { startAudio } = useAudioSettings();
    const { canRead, cta } = useBookStatus(book);
    
    useEffect(() => { 
        setBook(null);
        mockApi.fetchBookDetails(bookId).then(setBook);
    }, [bookId]);

    const handleCTA = async () => {
        // Start audio context on the first meaningful interaction
        await startAudio();

        if (canRead) {
            if (user.libraryBookIds.includes(bookId)) {
                setPage('reader');
            } else {
                addToLibrary(bookId);
            }
        } else {
            alert(`Action: ${cta}`);
        }
    };

    if (!book) return <BookDetailsSkeleton />;

    return (
        <div className="p-4 md:p-6">
            <div className="max-w-4xl mx-auto md:flex md:space-x-8">
                <div className="md:w-1/3 mb-6 md:mb-0">
                    <img src={book.cover} alt={book.title} className="rounded-lg shadow-lg w-full"/>
                </div>
                <div className="md:w-2/3 text-[var(--color-text-primary)]">
                    <h1 className="text-3xl md:text-4xl font-bold">{book.title}</h1>
                    <p className="text-lg text-[var(--color-text-muted)] mt-1">{book.author}</p>
                    <button onClick={handleCTA} className="w-full mt-6 py-3 rounded-lg bg-[var(--color-accent)] text-white font-bold text-lg hover:bg-[var(--color-accent-hover)] transition-colors">
                        {cta}
                    </button>
                    <div className="mt-8">
                        <h2 className="text-xl font-bold border-b border-[var(--color-border)] pb-2 mb-2">Synopsis</h2>
                        <p className="text-[var(--color-text-muted)] leading-relaxed">{book.synopsis}</p>
                    </div>
                </div>
            </div>
        </div>
    );
}
export default BookDetailsPage;