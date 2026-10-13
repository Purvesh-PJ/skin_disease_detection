import * as React from "react";
import { cn } from "../../lib/utils";

const Spinner = ({ size = "md", color, className, ...props }) => {
  const sizeClasses = {
    sm: "h-4 w-4 border-2",
    md: "h-6 w-6 border-[3px]",
    lg: "h-10 w-10 border-4",
  };

  return (
    <div
      className={cn(
        "rounded-full border-neutral-200 dark:border-neutral-700 border-t-emerald-600 dark:border-t-emerald-400 animate-spin shrink-0",
        sizeClasses[size] || sizeClasses.md,
        className
      )}
      style={color ? { borderTopColor: color } : undefined}
      {...props}
    />
  );
};

export { Spinner };
export default Spinner;
