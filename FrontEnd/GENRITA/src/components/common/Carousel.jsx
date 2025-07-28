// --- FILE: src/components/common/Carousel.jsx (REVISED) ---
import React from 'react';

function Carousel({ title, children }) {
    return (
        <section className="mb-10">
            {/* Padding is now inherited from the parent container in HomePage */}
            <h2 className="text-2xl font-bold text-[var(--color-text-primary)] mb-4">{title}</h2>
            <div className="flex space-x-4 overflow-x-auto pb-4 -mb-4">
                {children}
            </div>
        </section>
    );
}
export default Carousel;