// --- FILE: src/components/common/MultiSelectDropdown.jsx (NEW FILE) ---
// A new, reusable, and user-friendly multi-select component.
import React, { useState, useRef, useEffect } from 'react';

const MultiSelectDropdown = ({ label, options, selectedIds, onChange }) => {
    const [isOpen, setIsOpen] = useState(false);
    const wrapperRef = useRef(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [wrapperRef]);

    const handleSelect = (id) => {
        const newSelectedIds = selectedIds.includes(id)
            ? selectedIds.filter(selectedId => selectedId !== id)
            : [...selectedIds, id];
        onChange(newSelectedIds);
    };

    const selectedLabels = selectedIds.map(id => {
        const entry = Object.entries(options).find(([_, value]) => value === id);
        return entry ? entry[0] : '';
    }).filter(Boolean);

    return (
        <div className="relative" ref={wrapperRef}>
            <label className="block text-sm font-medium text-[var(--color-text-primary)] mb-1">{label}</label>
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className="w-full text-left p-2 bg-[var(--color-background)] border border-[var(--color-border)] rounded-md"
            >
                <span className="block truncate">
                    {selectedLabels.length > 0 ? selectedLabels.join(', ') : 'Select...'}
                </span>
            </button>
            {isOpen && (
                <div className="absolute z-10 mt-1 w-full bg-[var(--color-background-secondary)] shadow-lg border border-[var(--color-border)] rounded-md max-h-60 overflow-auto">
                    <ul className="py-1">
                        {Object.entries(options).map(([name, id]) => (
                            <li
                                key={id}
                                onClick={() => handleSelect(id)}
                                className="px-3 py-2 text-sm text-[var(--color-text-primary)] cursor-pointer hover:bg-[var(--color-background)] flex items-center"
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedIds.includes(id)}
                                    readOnly
                                    className="mr-2"
                                />
                                {name}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default MultiSelectDropdown;
