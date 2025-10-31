// --- FILE: src/components/common/SearchFilters.jsx (REVISED) ---
import React from 'react';

// Based on Swagger enum: SearchResultOrder
const sortOptions = [
    { value: 0, label: 'Oldest First' },
    { value: 1, label: 'Newest First' },
    { value: 2, label: 'Random' },
    { value: 3, label: 'Most Likes' },
    { value: 4, label: 'Users Have' },
    { value: 5, label: 'A-Z' },
    { value: 6, label: 'Z-A' },
];

// Based on Swagger enum: SearchMode
const searchModeOptions = [
    { value: 0, label: 'All Fields' },
    { value: 1, label: 'Title' },
    { value: 2, label: 'Author' },
    { value: 3, label: 'Category' },
    { value: 4, label: 'Synopsis' },
];

const SearchFilters = ({ filters, onFilterChange }) => {
    const handleChange = (e) => {
        const { name, value } = e.target;
        // Convert to number if it's a numeric field
        const processedValue = name === 'order' || name === 'searchMode' ? parseInt(value, 10) : value;
        onFilterChange({ ...filters, [name]: processedValue });
    };

    return (
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-end mb-4 gap-4">
            {/* Search Mode Filter */}
            <div className="flex items-center">
                <label htmlFor="search-mode" className="text-sm font-medium text-[var(--color-text-muted)] mr-2 whitespace-nowrap">
                    Search In:
                </label>
                <select
                    id="search-mode"
                    name="searchMode"
                    value={filters.searchMode}
                    onChange={handleChange}
                    className="w-full bg-[var(--color-background-secondary)] border border-[var(--color-border)] rounded-md text-sm p-2 focus:ring-[var(--color-accent)] focus:border-[var(--color-accent)]"
                >
                    {searchModeOptions.map(option => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
            </div>

            {/* Published Date Filter */}
             <div className="flex items-center">
                <label htmlFor="published-date" className="text-sm font-medium text-[var(--color-text-muted)] mr-2 whitespace-nowrap">
                    Published After:
                </label>
                <input
                    type="date"
                    id="published-date"
                    name="publishedDate"
                    value={filters.publishedDate}
                    onChange={handleChange}
                    className="w-full bg-[var(--color-background-secondary)] border border-[var(--color-border)] rounded-md text-sm p-2 focus:ring-[var(--color-accent)] focus:border-[var(--color-accent)]"
                />
            </div>

            {/* Sort Order Filter */}
            <div className="flex items-center">
                <label htmlFor="sort-order" className="text-sm font-medium text-[var(--color-text-muted)] mr-2 whitespace-nowrap">
                    Sort by:
                </label>
                <select
                    id="sort-order"
                    name="order"
                    value={filters.order}
                    onChange={handleChange}
                    className="w-full bg-[var(--color-background-secondary)] border border-[var(--color-border)] rounded-md text-sm p-2 focus:ring-[var(--color-accent)] focus:border-[var(--color-accent)]"
                >
                    {sortOptions.map(option => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
            </div>
        </div>
    );
};
export default SearchFilters;

