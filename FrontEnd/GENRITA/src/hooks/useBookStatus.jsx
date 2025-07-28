// --- FILE: src/hooks/useBookStatus.js ---
import { useAuth } from '../contexts/AuthContext';
export const useBookStatus = (book) => {
    const { user } = useAuth();
    if (!user || !book) return { canRead: false, statusLabel: '', cta: '' };

    const isOwned = user.ownedBookIds.includes(book.id);
    const inLibrary = user.libraryBookIds.includes(book.id);
    const hasActiveSub = user.subscription.status === 'active';

    if (isOwned || book.access === 'free') {
        return { canRead: true, statusLabel: 'Read Now', cta: 'Read Now' };
    }
    if (book.access === 'subscription') {
        if (hasActiveSub) {
            return { canRead: true, statusLabel: 'Subscription', cta: inLibrary ? 'Read Now' : 'Add to Library' };
        } else {
            return { canRead: false, statusLabel: 'Subscription Expired', cta: 'Renew Subscription' };
        }
    }
    if (book.access === 'purchase') {
        return { canRead: false, statusLabel: 'Purchase', cta: 'Buy to Read' };
    }
    return { canRead: false, statusLabel: '', cta: '' };
};