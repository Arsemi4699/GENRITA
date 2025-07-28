// --- FILE: src/components/common/SkeletonLoader.js ---
const CarouselSkeleton = () => (
    <div className="mb-10">
        <div className="h-8 w-1/3 bg-[var(--color-background-secondary)] rounded mb-4"></div>
        <div className="flex space-x-4">
            {[...Array(5)].map((_, i) => (
                <div key={i} className="w-36 md:w-40 flex-shrink-0">
                    <div className="bg-[var(--color-background-secondary)] aspect-[2/3] rounded-lg"></div>
                    <div className="h-4 mt-2 bg-[var(--color-background-secondary)] rounded w-3/4"></div>
                    <div className="h-3 mt-1 bg-[var(--color-background-secondary)] rounded w-1/2"></div>
                </div>
            ))}
        </div>
    </div>
);

export const HomePageSkeleton = () => {
    return (
        <div className="py-6 animate-pulse">
            <div className="mb-10">
                <div className="bg-[var(--color-background-secondary)] h-[200px] rounded-lg"></div>
            </div>
            {/* Render three carousels to match the final layout */}
            <CarouselSkeleton />
            <CarouselSkeleton />
            <CarouselSkeleton />
        </div>
    );
}

export const BookDetailsSkeleton = () => {
    return (
        <div className="p-4 md:p-6 animate-pulse">
            <div className="max-w-4xl mx-auto md:flex md:space-x-8">
                <div className="md:w-1/3 mb-6 md:mb-0">
                    <div className="aspect-[2/3] bg-[var(--color-background-secondary)] rounded-lg"></div>
                </div>
                <div className="md:w-2/3">
                    <div className="h-10 bg-[var(--color-background-secondary)] rounded w-3/4 mb-2"></div>
                    <div className="h-6 bg-[var(--color-background-secondary)] rounded w-1/2 mb-6"></div>
                    <div className="h-14 bg-[var(--color-background-secondary)] rounded-lg w-full mb-8"></div>
                    <div className="h-6 bg-[var(--color-background-secondary)] rounded w-1/4 mb-4"></div>
                    <div className="space-y-2">
                        <div className="h-4 bg-[var(--color-background-secondary)] rounded w-full"></div>
                        <div className="h-4 bg-[var(--color-background-secondary)] rounded w-full"></div>
                        <div className="h-4 bg-[var(--color-background-secondary)] rounded w-5/6"></div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// --- FILE: src/components/common/SkeletonLoader.js ---
export const SkeletonLoader = ({ type = 'card', count = 1 }) => {
    const Card = () => <div className="animate-pulse"><div className="bg-gray-300 dark:bg-gray-700 rounded-lg aspect-[2/3]"></div><div className="h-4 mt-2 bg-gray-300 dark:bg-gray-700 rounded w-3/4"></div><div className="h-3 mt-1 bg-gray-300 dark:bg-gray-700 rounded w-1/2"></div></div>;
    if (type === 'card') return <>{Array.from({ length: count }, (_, i) => <Card key={i} />)}</>;
    return <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-full animate-pulse"></div>;
}