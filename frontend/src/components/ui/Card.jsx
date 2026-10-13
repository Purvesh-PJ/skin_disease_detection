import * as React from "react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const cardVariants = cva(
  "rounded-xl border transition-all text-neutral-900 dark:text-neutral-100",
  {
    variants: {
      variant: {
        elevated: "bg-white dark:bg-[#181818] border-neutral-200 dark:border-neutral-800 shadow-sm",
        outlined: "bg-white dark:bg-[#181818] border-neutral-200 dark:border-neutral-800 shadow-none",
        subtle: "bg-neutral-50 dark:bg-[#1c1c1c] border-neutral-100 dark:border-neutral-800/60 shadow-none",
        flat: "bg-neutral-50 dark:bg-[#1c1c1c] border-transparent shadow-none",
        interactive:
          "bg-white dark:bg-[#181818] border-neutral-200 dark:border-neutral-800 shadow-sm hover:border-emerald-500/50 hover:shadow-brand-glow hover:-translate-y-0.5 cursor-pointer",
      },
    },
    defaultVariants: {
      variant: "elevated",
    },
  }
);

const Card = React.forwardRef(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(cardVariants({ variant }), className)}
    {...props}
  />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
  // eslint-disable-next-line jsx-a11y/heading-has-content
  <h3
    ref={ref}
    className={cn("text-lg font-semibold leading-none tracking-tight", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-neutral-500 dark:text-neutral-400", className)}
    {...props}
  />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  />
));
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardFooter, CardTitle, CardDescription, CardContent };
export default Card;
