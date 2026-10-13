import { Link } from 'react-router-dom';
import { Button } from '../../components/ui';
import { ROUTES } from '../../constants';

const NotFound = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-neutral-50 dark:bg-[#0a0a0a] text-center select-none">
      <h1 className="text-8xl font-black text-emerald-600 dark:text-emerald-500 m-0 mb-2 tracking-tight">
        404
      </h1>
      <p className="text-lg text-neutral-600 dark:text-neutral-400 mb-6 font-medium">
        Page not found
      </p>
      <Link to={ROUTES.HOME} className="no-underline">
        <Button size="lg">Go Home</Button>
      </Link>
    </div>
  );
};

export default NotFound;
