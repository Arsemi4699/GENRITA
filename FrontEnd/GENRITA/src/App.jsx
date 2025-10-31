// --- FILE: src/App.jsx (REVISED) ---
import { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { AudioSettingsProvider } from './contexts/AudioSettingsContext';

// Reader Pages
import AuthPage from './pages/AuthPage';
import HomePage from './pages/HomePage';
import SearchPage from './pages/SearchPage';
import LibraryPage from './pages/LibraryPage';
import ProfilePage from './pages/ProfilePage';
import BookDetailsPage from './pages/BookDetailsPage';
import ReaderPage from './pages/ReaderPage';
import SubscriptionPage from './pages/SubscriptionPage'; // New
import PaymentStatusPage from './pages/PaymentStatusPage'; // New

// Author Pages
import AuthorDashboardPage from './author/AuthorDashboardPage';
import NewBookPage from './author/NewBookPage';
import JobDetailsPage from './author/JobDetailsPage';
import AuthorProfilePage from './author/AuthorProfilePage'; // New author profile page

// Layout Components
import DesktopNav from './layout/DesktopNav';
import MobileHeader from './layout/MobileHeader';
import BottomTabNav from './layout/BottomTabNav';

function MainApp() {
    const { isAuthenticated, user, logout, refetchUser } = useAuth();

    const [page, setPage] = useState(null); // Start with null
    const [selectedBookId, setSelectedBookId] = useState(null);
    const [selectedJobId, setSelectedJobId] = useState(null);
    const [pageHistory, setPageHistory] = useState([]);

    const isAuthor = user?.role === 'author';

    // *** THE NAVIGATION FIX ***
    // This useEffect now ONLY runs when the user's authentication state changes.
    // It correctly sets the initial page without interfering with later navigation.
    useEffect(() => {
        if (isAuthenticated && user?.role) {
            const initialPage = isAuthor ? 'authorDashboard' : 'home';
            setPage(initialPage);
            setPageHistory([initialPage]);
        } else {
            // Reset state on logout
            setPage(null);
            setPageHistory([]);
        }
    }, [isAuthenticated, user, isAuthor]); // Removed 'page' from dependencies

    const navigateTo = (newPage) => {
        setPageHistory(prev => [...prev, newPage]);
        setPage(newPage);
    };

    const navigateBack = () => {
        if (pageHistory.length <= 1) return;
        const newHistory = pageHistory.slice(0, -1);
        setPageHistory(newHistory);
        setPage(newHistory[newHistory.length - 1]);
    };

    const renderPage = () => {
        if (!page) return null;

        if (isAuthor) {
            const authorProps = { setPage: navigateTo, setSelectedJobId };
            switch (page) {
                case 'authorDashboard': return <AuthorDashboardPage {...authorProps} />;
                case 'newBook': return <NewBookPage {...authorProps} />;
                case 'jobDetails': return <JobDetailsPage jobId={selectedJobId} {...authorProps} />;
                case 'authorProfile': return <AuthorProfilePage onLogout={logout} />; // New route for author profile
                default: return <AuthorDashboardPage {...authorProps} />;
            }
        }

        const readerProps = { setPage: navigateTo, setSelectedBookId };
        switch (page) {
            case 'home': return <HomePage {...readerProps} />;
            case 'search': return <SearchPage {...readerProps} />;
            case 'library': return <LibraryPage {...readerProps} />;
            case 'profile': return <ProfilePage onLogout={logout} setPage={navigateTo} />; // Pass setPage
            case 'bookDetails': return <BookDetailsPage bookId={selectedBookId} {...readerProps} />;
            case 'reader': return <ReaderPage bookId={selectedBookId} setPage={navigateTo} />;
            case 'subscription': return <SubscriptionPage setPage={navigateTo} />;
            case 'paymentStatus': return <PaymentStatusPage setPage={navigateTo} refetchUser={refetchUser} />;
            default: return <HomePage {...readerProps} />;
        }
    };

    if (!isAuthenticated) return <AuthPage />;

    // Navigation rendering is now cleaner and role-based
    const renderNav = () => {
        if (page === 'subscription' || page === 'paymentStatus') return null; // Hide nav on payment pages

        if (isAuthor) {
            // Author gets a consistent header with navigation
            return (
                <nav className="bg-[var(--color-background-secondary)] shadow-md sticky top-0 z-20 flex items-center justify-between px-6 py-3 border-b border-[var(--color-border)]">
                    <h1 className="text-2xl font-bold text-[var(--color-accent)]">Author Portal</h1>
                    <div className="flex items-center gap-6">
                        <button onClick={() => navigateTo('authorDashboard')} className={`text-sm font-medium ${page === 'authorDashboard' ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-primary)]'}`}>Dashboard</button>
                        <button onClick={() => navigateTo('authorProfile')} className={`text-sm font-medium ${page === 'authorProfile' ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-primary)]'}`}>Profile</button>
                        <button onClick={logout} className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">Logout</button>
                    </div>
                </nav>
            );
        }

        // Reader navigation remains the same
        const getMobileHeader = () => {
            if (page === 'bookDetails') return <MobileHeader title="Details" onBack={navigateBack} />;
            const mainPages = ['home', 'search', 'library', 'profile'];
            if (mainPages.includes(page)) return <MobileHeader title={page === 'home' ? 'Bookworm' : page.charAt(0).toUpperCase() + page.slice(1)} />;
            return null;
        };

        return (
            <>
                {page !== 'reader' && <DesktopNav activePage={page} setPage={navigateTo} onLogout={logout} />}
                {page !== 'reader' && getMobileHeader()}
                {page !== 'reader' && <BottomTabNav activePage={page} setPage={navigateTo} />}
            </>
        );
    };

    return (
        <div className="bg-[var(--color-background)] text-[var(--color-text-primary)] min-h-screen transition-colors duration-300">
            {renderNav()}
            <main className="main-content pb-20 md:pb-0">{renderPage()}</main>
        </div>
    );
}

export default function App() {
    return (
        <ThemeProvider>
            <AuthProvider>
                <AudioSettingsProvider>
                    <MainApp />
                </AudioSettingsProvider>
            </AuthProvider>
        </ThemeProvider>
    );
}