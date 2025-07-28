// --- FILE: src/pages/LibraryPage.js ---
import { useAuth } from '../contexts/AuthContext';
import { mockBooks } from '../data/mockData';
import BookCard from '../components/common/BookCard';
const LibraryPage = ({ setPage, setSelectedBookId }) => {
    const { user } = useAuth();
    const viewDetails = (bookId) => { setSelectedBookId(bookId); setPage('bookDetails'); };
    if (!user) return <div>Loading...</div>; // Could be a skeleton too
    const libraryBooks = user.libraryBookIds.map(id => mockBooks[id]);
    return (
        <div className="p-4 md:p-6">
            {libraryBooks.length === 0 ? <p className="text-center text-[var(--color-text-muted)] mt-8">Your library is empty.</p> : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-x-4 gap-y-8">
                    {libraryBooks.map(b => b && <BookCard key={b.id} book={b} onClick={() => viewDetails(b.id)} />)}
                </div>
            )}
        </div>
    );
}
export default LibraryPage;