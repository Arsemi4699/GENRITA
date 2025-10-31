// --- FILE: src/author/JobDetailsPage.jsx ---
import React, { useState, useEffect, useCallback } from 'react';
import { authorApi } from '../services/authorApi';

const POLL_INTERVAL = 3000; // 3 seconds

const JobDetailsPage = ({ jobId, setPage }) => {
    const [job, setJob] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchStatus = useCallback(async () => {
        if (!jobId) {
            setError("No Job ID provided.");
            setIsLoading(false);
            return;
        }

        console.log(`[Polling] Checking job status for ID: ${jobId}...`);

        try {
            const data = await authorApi.getJobStatus(jobId);
            console.log("[Polling] Response:", data);
            setJob(data);
            setError(null);
        } catch (err) {
            console.error("[Polling] Error fetching status:", err);
            setError(err.message || "Failed to fetch job status.");
        } finally {
            setIsLoading(false);
        }
    }, [jobId]);

    useEffect(() => {
        if (!jobId) return;

        // Initial fetch
        fetchStatus();

        // Set up polling interval
        const intervalId = setInterval(() => {
            fetchStatus();
        }, POLL_INTERVAL);

        // Cleanup on unmount
        return () => clearInterval(intervalId);
    }, [fetchStatus, jobId]);

    const renderStatus = () => {
        if (!job) return null;

        const status = job.status?.toLowerCase?.() || "unknown";
        const baseClasses = "text-lg font-semibold";

        switch (status) {
            case "queued":
                return (
                    <p className={`${baseClasses} text-yellow-600`}>
                        Your book is in the queue to be processed.
                    </p>
                );
            case "processing":
                return (
                    <p className={`${baseClasses} text-blue-600 animate-pulse`}>
                        AI is processing your book...
                    </p>
                );
            case "completed":
                return (
                    <p className={`${baseClasses} text-green-600`}>
                        Processing complete! The book is now available in the reader app.
                    </p>
                );
            case "failed":
                return (
                    <p className={`${baseClasses} text-red-600`}>
                        Processing failed: {job.errorMessage || "Unknown error."}
                    </p>
                );
            default:
                return (
                    <p className={`${baseClasses} text-gray-500`}>
                        Unknown status: {job.status}
                    </p>
                );
        }
    };

    if (isLoading) return <div className="p-6 text-center">Loading job details...</div>;

    return (
        <div className="p-4 md:p-6 max-w-3xl mx-auto text-center">
            <button
                onClick={() => setPage("authorDashboard")}
                className="text-sm font-semibold text-[var(--color-accent)] hover:underline mb-6"
            >
                &larr; Back to Dashboard
            </button>

            <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">
                Processing Status
            </h1>
            <p className="text-sm text-[var(--color-text-muted)] mb-8">
                Job ID: {jobId}
            </p>

            <div className="bg-[var(--color-background-secondary)] p-8 rounded-lg shadow min-h-[150px] flex items-center justify-center">
                {error ? (
                    <p className="text-lg font-semibold text-red-600">{error}</p>
                ) : (
                    renderStatus()
                )}
            </div>
        </div>
    );
};

export default JobDetailsPage;
