import type { CostCategory, IncomeCategory, MonthLedgerView } from "@airace/protocol";

// The money's categories in plain words (R7), shared by the Money screen and its windows.

export const INCOME_NAMES: Record<IncomeCategory, string> = {
  personal: "Personal plans",
  business: "Business plans",
};

export const COST_NAMES: Record<CostCategory, string> = {
  training: "Training",
  serving: "Serving customers",
  dataChecks: "Fact-checking data",
};

export const MONEY_WHAT: Record<IncomeCategory | CostCategory, string> = {
  personal: "What people paid each month for the personal plan.",
  business: "What businesses paid each month for their seats.",
  training: "What training your AIs cost, booked when each run started.",
  serving:
    "Running your AIs for every customer, free ones too. It grows with customers and AI size.",
  dataChecks: "Fact-checking the reading of runs that asked for it.",
};

export const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

export const profitOf = (m: MonthLedgerView) => total(m.income) - total(m.costs);
