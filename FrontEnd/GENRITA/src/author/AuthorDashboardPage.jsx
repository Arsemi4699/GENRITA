import React, { useState, useEffect } from 'react';
import { authorApi } from '../services/authorApi';

const AuthorDashboardPage = ({ setPage, setSelectedJobId }) => {
    const [jobs, setJobs] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [deletingId, setDeletingId] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        setIsLoading(true);
        authorApi.listAuthorBooks()
            .then(data => setJobs(data))
            .catch(err => {
                console.error("Failed to fetch author books:", err);
                setError("Could not load your books. Please try again later.");
            })
            .finally(() => setIsLoading(false));
    }, []);

    const viewJobDetails = (jobId) => {
        setSelectedJobId(jobId);
        setPage('jobDetails');
    };

    const deleteBook = async (jobbook) => {
        if (!window.confirm("Are you sure you want to delete this book?")) return;
        setDeletingId(jobbook.jobId);
        try {
            await authorApi.deleteBook(jobbook.bookId);
            setJobs(prev => prev.filter(job => job.jobId !== jobbook.jobId));
        } catch (err) {
            console.error("Failed to delete book:", err);
            alert("Failed to delete the book. Please try again.");
        } finally {
            setDeletingId(null);
        }
    };

    const getStatusChip = (status) => {
        const styles = {
            completed: 'bg-green-100 text-green-800',
            processing: 'bg-blue-100 text-blue-800 animate-pulse',
            failed: 'bg-red-100 text-red-800',
        };
        const formattedStatus = status.charAt(0).toUpperCase() + status.slice(1);
        return <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status] || ''}`}>{formattedStatus}</span>;
    };

    if (isLoading) return <div className="p-6 text-center">Loading dashboard...</div>;
    if (error) return <div className="p-6 text-center text-red-500">{error}</div>;
    console.log(jobs);
    return (
        <div className="max-w-6xl mx-auto p-4 md:p-6">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold text-[var(--color-text-primary)]">Author Dashboard</h1>
                <button
                    onClick={() => setPage('newBook')}
                    className="px-4 py-2 font-semibold text-white bg-[var(--color-accent)] rounded-lg hover:bg-[var(--color-accent-hover)]"
                >
                    + New Book
                </button>
            </div>

            {/* Horizontal scroll wrapper for better mobile UX */}
            <div className="overflow-x-auto rounded-lg shadow bg-[var(--color-background-secondary)]">
                {jobs.length > 0 ? (
                    <table className="min-w-full divide-y divide-[var(--color-border)]">
                        <thead className="bg-gray-50 dark:bg-gray-700">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Book Title</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Job ID</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Status</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Date Created</th>
                                <th className="relative px-6 py-3 text-right text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[var(--color-border)]">
                            {jobs.map(job => (
                                <tr key={job.jobId} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-[var(--color-text-primary)]">{job.bookTitle}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-[var(--color-text-muted)] font-mono">{job.jobId}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm">{getStatusChip(job.status)}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-[var(--color-text-muted)]">{new Date(job.createdAt).toLocaleDateString()}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium flex justify-end gap-3">
                                        <button
                                            onClick={() => viewJobDetails(job.jobId)}
                                            className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)]"
                                        >
                                            View
                                        </button>
                                        <button
                                            onClick={() => deleteBook(job)}
                                            disabled={deletingId === job.jobId}
                                            className={`text-red-600 hover:text-red-800 flex items-center gap-1 ${deletingId === job.jobId ? 'opacity-50 cursor-not-allowed' : ''
                                                }`}
                                        >
                                            {deletingId === job.jobId ? (
                                                // Spinner
                                                <svg
                                                    className="animate-spin h-4 w-4 text-red-600"
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    fill="none"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <circle
                                                        className="opacity-25"
                                                        cx="12"
                                                        cy="12"
                                                        r="10"
                                                        stroke="currentColor"
                                                        strokeWidth="4"
                                                    ></circle>
                                                    <path
                                                        className="opacity-75"
                                                        fill="currentColor"
                                                        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                                                    ></path>
                                                </svg>
                                            ) : (
                                                'Delete'
                                            )}
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                ) : (
                    <p className="p-8 text-center text-[var(--color-text-muted)]">
                        You haven't processed any books yet.
                    </p>
                )}
            </div>
        </div>
    );
};

export default AuthorDashboardPage;
