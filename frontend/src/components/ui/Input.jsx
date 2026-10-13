import * as React from "react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

const inputVariants = cva(
  "flex w-full rounded-lg border bg-white dark:bg-[#181818] text-neutral-900 dark:text-neutral-100 transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-neutral-400 dark:placeholder:text-neutral-500 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
  {
    variants: {
      size: {
        sm: "h-9 px-3 text-xs rounded-md",
        md: "h-11 px-4 py-2 text-sm rounded-lg",
        lg: "h-12 px-4.5 text-base rounded-xl",
      },
      error: {
        true: "border-red-500 focus-visible:ring-2 focus-visible:ring-red-500/20",
        false: "border-neutral-200 dark:border-neutral-800 focus-visible:border-emerald-500 focus-visible:ring-2 focus-visible:ring-emerald-500/20",
      },
    },
    defaultVariants: {
      size: "md",
      error: false,
    },
  }
);

const Input = React.forwardRef(({ className, type, size, error, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(inputVariants({ size, error: Boolean(error), className }))}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";

export { Input };
export default Input;
