// --- FILE: src/components/layout/MobileHeader.js ---
import { BackIcon } from '../components/common/icons/index';
const MobileHeader = ({ title, onBack }) => {
    return (
        <div className="md:hidden sticky top-0 z-20 flex items-center justify-between p-4 bg-[var(--color-background-secondary)] border-b border-[var(--color-border)]">
            {onBack ? (
                <button onClick={onBack} className="p-1 -ml-1 text-[var(--color-text-primary)]">
                    <BackIcon />
                </button>
            ) : (
                <div className="w-8"></div> // Placeholder for alignment
            )}
            <h1 className="text-lg font-bold text-[var(--color-text-primary)] absolute left-1/2 -translate-x-1/2">
                {title}
            </h1>
            <div className="w-8"></div> {/* Placeholder for alignment */}
        </div>
    );
}
export default MobileHeader;