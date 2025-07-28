// --- FILE: src/components/layout/BottomTabNav.js ---
import { HomeIcon, SearchIcon, LibraryIcon, UserIcon } from '../components/common/icons/index'
const BottomTabNav = ({ activePage, setPage }) => {
    return (
        <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-[var(--color-background-secondary)] border-t border-[var(--color-border)] z-20">
            <div className="flex justify-around">
                {[{ id: 'home', icon: <HomeIcon /> }, { id: 'search', icon: <SearchIcon /> }, { id: 'library', icon: <LibraryIcon /> }, { id: 'profile', icon: <UserIcon /> }].map(item => (
                    <button key={item.id} onClick={() => setPage(item.id)} className={`flex flex-col items-center justify-center w-full pt-2 pb-1 text-sm transition-colors ${activePage === item.id ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)]'}`}>
                        {item.icon}
                        <span className="mt-1 capitalize">{item.id}</span>
                    </button>
                ))}
            </div>
        </nav>
    );
}
export default BottomTabNav;