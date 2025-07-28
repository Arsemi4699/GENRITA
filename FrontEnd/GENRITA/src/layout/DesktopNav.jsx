// --- FILE: src/components/layout/DesktopNav.js ---
import { useAuth } from "../contexts/AuthContext";
const DesktopNav = ({ activePage, setPage, onLogout }) => {
    const { user } = useAuth();
    return (
        <nav className="hidden md:flex bg-[var(--color-background-secondary)] shadow-md sticky top-0 z-20 items-center justify-between px-6 py-3 border-b border-[var(--color-border)]">
            <div className="flex items-center space-x-6">
                <h1 className="text-2xl font-bold text-[var(--color-accent)]">Bookworm</h1>
                {[{ id: 'home', label: 'Home' }, { id: 'search', label: 'Search' }, { id: 'library', label: 'My Library' }].map(item => (
                    <button key={item.id} onClick={() => setPage(item.id)} className={`px-3 py-2 rounded-md text-sm font-medium transition-colors text-[var(--color-text-primary)] hover:bg-[var(--color-background)] ${activePage === item.id && 'bg-[var(--color-background)]'}`}>
                        {item.label}
                    </button>
                ))}
            </div>
            <div className="flex items-center space-x-4">
                <button onClick={() => setPage('profile')} className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-[var(--color-accent)] rounded-full flex items-center justify-center text-white font-bold">{user.name.charAt(0)}</div>
                    <span className="text-sm font-medium text-[var(--color-text-primary)]">{user.name}</span>
                </button>
                <button onClick={onLogout} className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">
                    Logout
                </button>
            </div>
        </nav>
    );
}
export default DesktopNav;