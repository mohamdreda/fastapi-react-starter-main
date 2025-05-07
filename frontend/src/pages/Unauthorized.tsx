export default function Unauthorized() {
    return (
      <div className="container mx-auto p-8 text-center">
        <h1 className="text-3xl font-bold text-red-600 mb-4">403 - Unauthorized</h1>
        <p className="text-lg text-gray-600 dark:text-gray-300">
          You don't have permission to access this page
        </p>
      </div>
    );
  }