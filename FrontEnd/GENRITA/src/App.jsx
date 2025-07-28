// --- FILE: src/App.js ---
import { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import AuthPage from './pages/AuthPage';
import HomePage from './pages/HomePage';
import SearchPage from './pages/SearchPage';
import LibraryPage from './pages/LibraryPage';
import ProfilePage from './pages/ProfilePage';
import BookDetailsPage from './pages/BookDetailsPage';
import ReaderPage from './pages/ReaderPage';
import DesktopNav from './layout/DesktopNav';
import MobileHeader from './layout/MobileHeader';
import BottomTabNav from './layout/BottomTabNav';
import { AudioSettingsProvider } from './contexts/AudioSettingsContext';


function MainApp() {
    const { isAuthenticated, logout } = useAuth();
    const [page, setPage] = useState('home');
    const [selectedBookId, setSelectedBookId] = useState(null);
    const [pageHistory, setPageHistory] = useState(['home']);

    const navigateTo = (newPage) => {
        setPageHistory(prev => [...prev, newPage]);
        setPage(newPage);
    };

    const navigateBack = () => {
        if (pageHistory.length <= 1) return;
        const historyCopy = [...pageHistory];
        historyCopy.pop();
        const previousPage = historyCopy[historyCopy.length - 1];
        setPageHistory(historyCopy);
        setPage(previousPage);
    };

    const renderPage = () => {
        const props = { setPage: navigateTo, setSelectedBookId };
        switch (page) {
            case 'home': return <HomePage {...props} />;
            case 'search': return <SearchPage {...props} />;
            case 'library': return <LibraryPage {...props} />;
            case 'profile': return <ProfilePage onLogout={logout} />;
            case 'bookDetails': return <BookDetailsPage bookId={selectedBookId} {...props} />;
            case 'reader': return <ReaderPage bookId={selectedBookId} setPage={navigateTo} />;
            default: return <HomePage {...props} />;
        }
    };

    if (!isAuthenticated) return <AuthPage />;

    const getMobileHeader = () => {
        const mainPages = ['home', 'search', 'library', 'profile'];
        if (mainPages.includes(page)) {
            return <MobileHeader title={page === 'home' ? 'Bookworm' : page.charAt(0).toUpperCase() + page.slice(1)} />;
        }
        if (page === 'bookDetails') {
            return <MobileHeader title="Details" onBack={navigateBack} />;
        }
        return null;
    };

    return (
        <div className="bg-[var(--color-background)] text-[var(--color-text-primary)] min-h-screen transition-colors duration-300">
            {page !== 'reader' && <DesktopNav activePage={page} setPage={navigateTo} onLogout={logout} />}
            {page !== 'reader' && getMobileHeader()}
            <main className="main-content pb-20 md:pb-0">{renderPage()}</main>
            {page !== 'reader' && <BottomTabNav activePage={page} setPage={navigateTo} />}
        </div>
    );
}

export default function App() {
    useEffect(() => {
        document.documentElement.dir = 'ltr';
        document.body.style.fontFamily = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif";
    }, []);
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