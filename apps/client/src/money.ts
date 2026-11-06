import type { CostCategory, IncomeCategory, MonthLedgerView } from "@airace/protocol";

// The money's categories in plain words (R7), shared by the Money screen and its windows.

export const INCOME_NAMES: Record<IncomeCategory, string> = {
  personal: "Personal plans",
  business: "Business plans",
  appRevenue: "Partner apps",
  investmentGains: "Investment gains",
};

export const COST_NAMES: Record<CostCategory, string> = {
  training: "Training",
  serving: "Serving customers",
  dataChecks: "Fact-checking data",
  marketing: "Ads and battles",
  dataCenterBuild: "Building data centers",
  dataCenterUpkeep: "Data center upkeep",
  interest: "Loan interest",
  events: "Keynotes and showcases",
  sponsorships: "Sponsorships",
  looks: "Looks",
  appCosts: "Partner app costs",
  investmentLosses: "Investment losses",
  overhead: "Office and support",
  team: "Team",
};

export const MONEY_WHAT: Record<IncomeCategory | CostCategory, string> = {
  personal: "What people paid each month for the personal plan.",
  business: "What businesses paid each month for their seats.",
  training: "What training your AIs cost, booked when each run started.",
  serving:
    "Running your AIs for every customer, free ones too. It grows with customers and AI size.",
  dataChecks: "Fact-checking the reading of runs that asked for it.",
  marketing: "What your ad campaigns spent, a day at a time, and each battle's $5,000 fee.",
  dataCenterBuild: "Building your own data centers, paid when building starts.",
  dataCenterUpkeep: "Running your data centers: power, staff and repairs.",
  interest: "Interest on an emergency loan, paid at each month's end.",
  appRevenue: "Your share of what your partner apps earned.",
  investmentGains: "What your investment fund gained at month ends. It stays in the fund.",
  events: "Launch keynotes and live showcases on stage.",
  sponsorships: "Putting your name on stadiums, conferences and festivals.",
  looks: "Orb styles and logo motions you bought.",
  appCosts: "Setting up partner apps, and their monthly running costs.",
  investmentLosses: "What your investment fund lost in a bad month.",
  team: "Your departments' monthly budgets (Research, Data team, Sales & marketing, Support & operations), paid a day at a time.",
  overhead: "Offices, support staff and admin. The bigger the company, the bigger the bill.",
};

export const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

export const profitOf = (m: MonthLedgerView) => total(m.income) - total(m.costs);
