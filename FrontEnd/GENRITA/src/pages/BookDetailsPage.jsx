// --- FILE: src/pages/BookDetailsPage.jsx (REVISED) ---
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useBookStatus } from '../hooks/useBookStatus';
import { BookDetailsSkeleton } from '../components/common/SkeletonLoader';
import { api } from '../services/api';
import { useAudioSettings } from '../contexts/AudioSettingsContext';

const BookDetailsPage = ({ bookId, setPage }) => {
    const [book, setBook] = useState(null);
    const { user, updateUserLibrary } = useAuth();
    const { startAudio } = useAudioSettings();
    const { canRead, cta, action } = useBookStatus(book);
    const inLibrary = user.libraryBookIds.includes(bookId);

    useEffect(() => {
        setBook(null);
        api.fetchBookDetails(bookId).then(setBook).catch(console.error);
    }, [bookId]);

    const handleCTA = async () => {
        await startAudio();
        if (action === 'read') {
            setPage('reader');
        } else if (action === 'add') {
            const res = await api.addToLibrary(bookId);
            updateUserLibrary(res.libraryBookIds);
        } else if (action === 'subscribe') {
            setPage('subscription');
        } else {
            console.log(`Unhandled Action: ${action}`);
        }
    };

    const handleRemoveFromLibrary = async () => {
        await api.removeFromLibrary(bookId);
        const updatedUser = await api.fetchMe(); // Refetch user to get updated library
        updateUserLibrary(updatedUser.libraryBookIds);
    };

    if (!book) return <BookDetailsSkeleton />;

    return (
        <div className="p-4 md:p-6">
            <div className="max-w-4xl mx-auto md:flex md:space-x-8">
                <div className="md:w-1/3 mb-6 md:mb-0">
                    <img src={book.cover} alt={book.title} className="rounded-lg shadow-lg w-full" />
                </div>
                <div className="md:w-2/3 text-[var(--color-text-primary)]">
                    <h1 className="text-3xl md:text-4xl font-bold">{book.title}</h1>
                    <p className="text-lg text-[var(--color-text-muted)] mt-1">{book.authorName}</p>
                    <button onClick={handleCTA} className="w-full mt-6 py-3 rounded-lg bg-[var(--color-accent)] text-white font-bold text-lg hover:bg-[var(--color-accent-hover)] transition-colors">
                        {cta}
                    </button>
                    {inLibrary && (
                        <button onClick={handleRemoveFromLibrary} className="w-full mt-2 py-3 rounded-lg bg-red-500/20 text-red-500 font-bold text-lg hover:bg-red-500/30 transition-colors">
                            Remove from Library
                        </button>
                    )}
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