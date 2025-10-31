// --- FILE: src/pages/SubscriptionPage.jsx (NEW FILE) ---
import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const PlanCard = ({ plan, onSelect, isLoading }) => (
    <div className="border border-[var(--color-border)] rounded-lg p-6 flex flex-col items-center text-center bg-[var(--color-background-secondary)] shadow transition-transform hover:-translate-y-1">
        <h3 className="text-xl font-bold text-[var(--color-accent)]">{plan.name}</h3>
        <p className="text-4xl font-extrabold my-4">${plan.price}<span className="text-base font-normal text-[var(--color-text-muted)]"> / {plan.duration} days</span></p>
        <p className="text-[var(--color-text-muted)] flex-grow">{plan.description}</p>
        <button
            onClick={() => onSelect(plan.id)}
            disabled={isLoading}
            className="w-full mt-6 py-2 rounded-lg bg-[var(--color-accent)] text-white font-bold hover:bg-[var(--color-accent-hover)] transition-colors disabled:opacity-50"
        >
            {isLoading ? 'Processing...' : 'Choose Plan'}
        </button>
    </div>
);


const SubscriptionPage = ({ setPage }) => {
    const [plans, setPlans] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isRedirecting, setIsRedirecting] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        api.getAllPlans()
            .then(data => setPlans(data))
            .catch(err => {
                console.error("Failed to fetch plans:", err);
                setError("Could not load subscription plans. Please try again later.");
            })
            .finally(() => setIsLoading(false));
    }, []);

    const handleSelectPlan = async (planId) => {
        setIsRedirecting(true);
        setError(null);
        try {
            const paymentUrl = await api.createPayment(planId);
            // The backend returns the payment URL. We redirect the user to it.
            if (paymentUrl && typeof paymentUrl === 'string') {
                window.location.href = paymentUrl;
            } else {
                setError("Could not initiate payment. No redirect URL received.");
                setIsRedirecting(false);
            }
        } catch (err) {
            setError(err.response?.data?.detail || "An error occurred while creating the payment request.");
            setIsRedirecting(false);
        }
    };

    if (isLoading) {
        return <div className="p-6 text-center">Loading subscription plans...</div>;
    }

    return (
        <div className="max-w-4xl mx-auto p-4 md:p-6 text-center">
            <button onClick={() => setPage('profile')} className="text-sm font-semibold text-[var(--color-accent)] hover:underline mb-6">
                &larr; Back to Profile
            </button>
            <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-2">Choose Your Plan</h1>
            <p className="text-[var(--color-text-muted)] mb-8">Unlock unlimited access to our subscription library.</p>

            {error && <p className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">{error}</p>}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {plans.length > 0 ? (
                    plans.map(plan => (
                        <PlanCard key={plan.id} plan={plan} onSelect={handleSelectPlan} isLoading={isRedirecting} />
                    ))
                ) : (
                    <p className="md:col-span-3 text-[var(--color-text-muted)]">No subscription plans are available at the moment.</p>
                )}
            </div>

            {isRedirecting && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-white p-6 rounded-lg text-center shadow-xl">
                        <p className="font-semibold">Redirecting to payment gateway...</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SubscriptionPage;