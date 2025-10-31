// --- FILE: src/hooks/useBookStatus.jsx (REVISED & FIXED) ---
import { useAuth } from '../contexts/AuthContext';

export const useBookStatus = (book) => {
    const { user } = useAuth();
    if (!user || !book) {
        return { canRead: false, statusLabel: 'Unavailable', cta: 'Unavailable', action: null };
    }
    console.log(user.subscription?.status);
    const inLibrary = user.libraryBookIds.includes(book.id);
    const hasActiveSub = user.subscription?.status === 'Active';

    // 1. Handle Free books first - they are always accessible.
    // Use .toLowerCase() to be safe against case variations like "Free" or "free".
    if (book.access?.toLowerCase() === 'free') {
        return {
            canRead: true,
            statusLabel: 'Free',
            cta: inLibrary ? 'Read Now' : 'Add to Library',
            action: inLibrary ? 'read' : 'add',
        };
    }

    // 2. Handle Subscription books
    if (book.access?.toLowerCase() === 'subscription') {
        console.log("need sub");
        if (hasActiveSub) {
            console.log("has sub");
            return {
                canRead: true,
                statusLabel: 'Subscription',
                cta: inLibrary ? 'Read Now' : 'Add to Library',
                action: inLibrary ? 'read' : 'add',
            };
        } else {
            console.log("no sub");
            return { canRead: false, statusLabel: 'Subscription', cta: 'Renew Subscription', action: 'subscribe' };
        }
    }

    // 3. Handle Purchasable books
    if (book.access?.toLowerCase() === 'purchase') {
        // If it's in the library, they own it.
        if (inLibrary) {
            return { canRead: true, statusLabel: 'In Library', cta: 'Read Now', action: 'read' };
        }
        // If not, they can add it (which implies buying in a real app)
        return { canRead: false, statusLabel: 'Purchase', cta: 'Add to Library', action: 'add' };
    }

    // 4. Default fallback for any other unhandled access types
    return { canRead: false, statusLabel: 'Unavailable', cta: 'Unavailable', action: null };
};