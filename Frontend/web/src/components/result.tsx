import React from "react";

export default function Result({ data }: any) {
  console.log(data);

  return (
    <div className="w-100 flex items-center justify-center px-4 py-8">
      <div className="bg-gray-900 border border-gray-800 rounded-xl shadow-lg overflow-hidden w-100 max-w-4xl">
        {data && (
          <div className="px-6 py-8 sm:px-10">
            <p className="text-center text-2xl font-semibold text-gray-300 mb-8">
              Potential Plant Issue
            </p>

            <div className="space-y-8">
              {/* Prediction */}
              <div className="text-center">
                <span className="inline-block bg-gradient-to-r from-green-400 to-blue-500 text-white rounded-full px-6 py-2 text-lg font-medium shadow">
                  {data.prediction}
                </span>
              </div>

              {/* Description */}
              <div>
                <h3 className="text-lg font-semibold text-gray-400 mb-2">Description</h3>
                <p className="text-gray-300 leading-relaxed">{data.cure_info.reason}</p>
              </div>

              {/* Treatment */}
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-2">Treatment</h3>
                <ol className="list-decimal pl-5 text-gray-300 leading-relaxed space-y-2">
                  {data.cure_info.treatment.steps.map((step: string, index: number) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>

              {/* Prevention */}
              <div>
                <h3 className="text-lg font-semibold text-yellow-400 mb-2">Prevention</h3>
                <ul className="list-disc pl-5 text-gray-300 leading-relaxed space-y-2">
                  {data.cure_info.prevention.steps.map((step: string, index: number) => (
                    <li key={index}>{step}</li>
                  ))}
                </ul>
              </div>

              {/* Alternate Images */}
              
                 <div>
                  <h3 className="text-lg font-semibold text-blue-400 mb-4">Alternate Images</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {data.cure_info.Image.map((imgUrl: string, index: number) => (
                      <img
                        key={index}
                        src={`http://127.0.0.1:5000/${imgUrl}`}
                        alt={`Alternate view ${index + 1}`}
                        className="w-full h-28 object-cover rounded-md border border-gray-700"
                      />
                    ))}
                  </div>
                </div> 
              
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
