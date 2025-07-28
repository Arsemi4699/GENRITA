// --- FILE: src/components/common/BookCard.js ---
import { useBookStatus } from '../../hooks/useBookStatus';
const BookCard = ({ book, onClick }) => {
    const { statusLabel } = useBookStatus(book);
    const labelStyles = {
        'Subscription': 'bg-yellow-400/80 text-yellow-900',
        'Purchase': 'bg-blue-400/80 text-blue-900',
        'Read Now': 'bg-green-400/80 text-green-900',
        'Subscription Expired': 'bg-red-400/80 text-red-900',
    };
    return (
        <div onClick={onClick} className="relative cursor-pointer group">
            <div className="overflow-hidden rounded-lg shadow-lg transition-transform duration-300 group-hover:-translate-y-1">
                <img src={book.cover} alt={`Cover for ${book.title}`} className="w-full h-auto aspect-[2/3] object-cover bg-gray-200" />
            </div>
            {statusLabel && <div className={`absolute top-2 right-2 text-xs font-bold px-2 py-1 rounded-full backdrop-blur-sm ${labelStyles[statusLabel] || 'bg-gray-400/80'}`}>{statusLabel}</div>}
            <h3 className="mt-2 text-sm font-semibold text-[var(--color-text-primary)] truncate">{book.title}</h3>
            <p className="text-xs text-[var(--color-text-muted)] truncate">{book.author}</p>
        </div>
    );
}
export default BookCard;